package io.github.guzinski.farol

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Recording
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import io.github.guzinski.farol.camera.DriveCamera
import io.github.guzinski.farol.camera.VideoTimeline
import io.github.guzinski.farol.camera.YuvFrame
import io.github.guzinski.farol.camera.thumbnail
import io.github.guzinski.farol.detect.Analysis
import io.github.guzinski.farol.detect.Backend
import io.github.guzinski.farol.detect.FrameGate
import io.github.guzinski.farol.detect.LiteRtRunner
import io.github.guzinski.farol.detect.ModelFile
import io.github.guzinski.farol.detect.ModelRunner
import io.github.guzinski.farol.detect.Pipeline
import io.github.guzinski.farol.i18n.Translator
import io.github.guzinski.farol.perception.DEFAULT_BAND
import io.github.guzinski.farol.replay.VideoFrameSource
import io.github.guzinski.farol.session.SessionRecorder
import io.github.guzinski.farol.speech.Speaker
import org.json.JSONObject
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class Settings(
    val use4k: Boolean = true,
    val exposure: Float = -1f,
    val meter: Boolean = false,
    val band: Double = DEFAULT_BAND,
    val speech: Boolean = true,
    val rate: Float = 1f,
    val language: String = "pt-BR",
    val backend: Backend = Backend.GPU,
    val model: ModelFile? = null,
)

enum class Operation { IDLE, STARTING, RECORDING, SAVING, REPLAYING }

class FarolController(private val activity: ComponentActivity) {
    private val context = activity.applicationContext
    private val preferences = context.getSharedPreferences("farol", 0)
    private val main = Handler(Looper.getMainLooper())
    private val mainExecutor = ContextCompat.getMainExecutor(activity)
    private val worker = Executors.newSingleThreadExecutor { task -> Thread(task, "farol-analysis") }
    private val cameraFuture = ProcessCameraProvider.getInstance(activity)
    private var provider: ProcessCameraProvider? = null
    private var camera: DriveCamera.Bound? = null
    private var permitted = false
    private var foreground = false
    private var closed = false
    private var cameraGeneration = 0
    private var modelGeneration = 0
    private var alertGeneration = 0
    private var exposureIndex = 0
    private var recording: Recording? = null
    private var recordingFailure: String? = null
    private var replayCancel: AtomicBoolean? = null

    val preview = PreviewView(activity).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
    }
    var models by mutableStateOf(ModelFile.list(context))
        private set
    var settings by mutableStateOf(Settings(
        use4k = preferences.getBoolean("4k", true),
        exposure = preferences.getFloat("ev", -1f),
        meter = preferences.getBoolean("meter", false),
        band = preferences.getFloat("band", DEFAULT_BAND.toFloat()).toDouble(),
        speech = preferences.getBoolean("speech", true),
        rate = preferences.getFloat("rate", 1f),
        language = preferences.getString("language", "pt-BR") ?: "pt-BR",
        backend = Backend.entries.firstOrNull { it.name == preferences.getString("backend", null) } ?: Backend.GPU,
        model = models.firstOrNull { it.file?.absolutePath == preferences.getString("model", "") }
            ?: models.firstOrNull { it.label == preferences.getString("model", "") }
            ?: preferredModel(),
    ))
        private set
    var modelReady by mutableStateOf(false)
        private set
    var modelBusy by mutableStateOf(false)
        private set
    var cameraReady by mutableStateOf(false)
        private set
    var operation by mutableStateOf(Operation.IDLE)
        private set
    var replay by mutableStateOf(false)
        private set
    var frame by mutableStateOf<Analysis?>(null)
        private set
    var videoFrame by mutableStateOf<Bitmap?>(null)
        private set
    var lastAlert by mutableStateOf<String?>(null)
        private set
    var cameraMessage by mutableStateOf<String?>(null)
        private set
    var modelMessage by mutableStateOf<String?>(null)
        private set
    var sessionMessage by mutableStateOf<String?>(null)
        private set
    var voiceMessage by mutableStateOf<String?>(null)
        private set
    private val speaker = Speaker(context) { voiceMessage = it }

    // The LiteRT delegate, pipeline state and session writers belong to this one worker.
    private var runner: ModelRunner? = null
    private var pipeline: Pipeline? = null
    private var liveTranslator = Translator.load(context.assets, settings.language)
    private var gate = FrameGate()
    private var previewOrigin: Long? = null
    private var drive: DriveLog? = null

    init {
        speaker.setLanguage(settings.language)
        cameraFuture.addListener({
            if (closed) return@addListener
            runCatching { cameraFuture.get() }.onSuccess {
                provider = it
                bindCamera()
            }.onFailure { cameraMessage = failure("Câmera indisponível", it) }
        }, mainExecutor)
        loadModel()
    }

    private fun preferredModel() = models.firstOrNull { it.name == "yolo11s.tflite" } ?: models.firstOrNull()

    fun refreshModels() {
        models = ModelFile.list(context)
        if (settings.model !in models) change(settings.copy(model = preferredModel()))
        else if (!modelReady && !modelBusy) loadModel()
    }

    fun change(updated: Settings) {
        if (operation != Operation.IDLE) return
        val old = settings
        settings = updated
        preferences.edit {
            putBoolean("4k", updated.use4k)
            putFloat("ev", updated.exposure)
            putBoolean("meter", updated.meter)
            putFloat("band", updated.band.toFloat())
            putBoolean("speech", updated.speech)
            putFloat("rate", updated.rate)
            putString("language", updated.language)
            putString("backend", updated.backend.name)
            putString("model", updated.model?.let { it.file?.absolutePath ?: it.label })
        }
        if (old.language != updated.language) speaker.setLanguage(updated.language)
        if (!updated.speech) speaker.stop()
        if (old.model != updated.model || old.backend != updated.backend) loadModel()
        else if (old.band != updated.band || old.language != updated.language) {
            worker.execute { resetPipeline(updated) }
            clearOverlay()
        }
        if (old.use4k != updated.use4k) bindCamera()
        if (old.exposure != updated.exposure) camera?.let { exposureIndex = it.setExposure(updated.exposure) }
        if (old.meter != updated.meter) updateMeter()
    }

    private fun loadModel() {
        val selection = settings
        val generation = ++modelGeneration
        modelReady = false
        modelBusy = selection.model != null
        modelMessage = if (selection.model == null) "Adicione um modelo .tflite em Android/media ou nos arquivos do app." else null
        clearOverlay()
        worker.execute {
            val result = runCatching {
                pipeline = null
                runner?.close()
                runner = null
                selection.model?.let { runner = LiteRtRunner(context, it, selection.backend) }
                resetPipeline(selection)
                runner != null
            }
            main.post {
                if (generation != modelGeneration || closed) return@post
                modelBusy = false
                modelReady = result.getOrDefault(false)
                result.exceptionOrNull()?.let { modelMessage = failure("Falha ao abrir o modelo", it) }
            }
        }
    }

    private fun resetPipeline(settings: Settings) {
        pipeline = runner?.let { Pipeline(it, settings.band) }
        liveTranslator = Translator.load(context.assets, settings.language)
        gate = FrameGate()
        previewOrigin = null
    }

    fun setForeground(active: Boolean) {
        foreground = active
        if (active) {
            bindCamera()
        } else {
            if (operation == Operation.STARTING || operation == Operation.RECORDING) stopDrive()
            if (operation == Operation.REPLAYING) stopReplay()
            speaker.stop()
            unbindCamera()
        }
    }

    fun permission(granted: Boolean) {
        permitted = granted
        if (granted) bindCamera()
    }

    private fun bindCamera() {
        val activeProvider = provider ?: return
        if (!permitted || !foreground || replay || closed) return
        if (camera != null && operation != Operation.IDLE) return
        unbindCamera()
        val generation = cameraGeneration
        val snapshot = settings
        worker.execute { resetPipeline(snapshot) }
        try {
            val bound = DriveCamera(context, activity, activeProvider, preview, worker).bind(snapshot.use4k) { image ->
                analyseLive(image, generation)
            }
            camera = bound
            exposureIndex = bound.setExposure(snapshot.exposure)
            updateMeter()
            cameraReady = true
            cameraMessage = null
        } catch (error: Exception) {
            cameraMessage = failure("Não foi possível iniciar a câmera", error)
        }
    }

    private fun unbindCamera() {
        cameraGeneration++
        camera?.analysis?.clearAnalyzer()
        camera = null
        cameraReady = false
        provider?.unbindAll()
        main.removeCallbacks(meterTask)
    }

    private val meterTask = object : Runnable {
        override fun run() {
            val bound = camera ?: return
            bound.meterOnLights(preview)
            main.postDelayed(this, 4_000)
        }
    }

    private fun updateMeter() {
        main.removeCallbacks(meterTask)
        if (settings.meter) meterTask.run() else camera?.camera?.cameraControl?.cancelFocusAndMetering()
    }

    private fun analyseLive(image: ImageProxy, generation: Int) {
        try {
            val processor = pipeline ?: return
            val timestamp = image.imageInfo.timestamp
            val log = drive
            val origin = if (log == null) {
                previewOrigin ?: timestamp.also { previewOrigin = it }
            } else log.timeline.startNs
            if (origin < 0 || timestamp + 1_000 < origin || !gate.accept(timestamp)) return
            val result = processor.analyse(YuvFrame.camera(image), (timestamp - origin).coerceAtLeast(0) / 1e9)
            if (log != null) {
                log.enqueue(timestamp, result)
                log.drain()
            }
            val last = result.alerts.lastOrNull()?.let(liveTranslator::alert)
            main.post {
                if (generation == cameraGeneration && foreground) present(result, last, speak = operation == Operation.RECORDING)
            }
        } catch (error: Exception) {
            pipeline = null
            main.post {
                if (generation != cameraGeneration) return@post
                modelReady = false
                modelMessage = failure("Falha na análise", error)
                recordingFailure = modelMessage
                if (operation == Operation.RECORDING || operation == Operation.STARTING) stopDrive()
            }
        } finally {
            image.close()
        }
    }

    @SuppressLint("MissingPermission")
    fun startDrive() {
        val bound = camera ?: return
        if (!modelReady || operation != Operation.IDLE) return
        val snapshot = settings
        var log: DriveLog? = null
        recordingFailure = null
        sessionMessage = null
        try {
            val timeline = bound.newTimeline()
            val session = SessionRecorder.create(context, metadata("drive", snapshot)
                .put("video_quality", bound.videoQuality).put("camera_id", bound.lens)
                .put("video_resolution", bound.video.resolutionInfo?.resolution?.let {
                    org.json.JSONArray(listOf(it.width, it.height))
                } ?: JSONObject.NULL)
                .put("analysis_resolution", bound.analysis.resolutionInfo?.resolution?.let {
                    org.json.JSONArray(listOf(it.width, it.height))
                } ?: JSONObject.NULL)
                .put("exposure_index", exposureIndex))
            val activeLog = DriveLog(session, timeline, Translator.load(context.assets, snapshot.language))
            log = activeLog
            worker.execute {
                resetPipeline(snapshot)
                drive = activeLog
            }
            clearOverlay()
            operation = Operation.STARTING
            recording = bound.video.output.prepareRecording(context, FileOutputOptions.Builder(session.video).build())
                .start(mainExecutor) { event ->
                    when (event) {
                        is VideoRecordEvent.Status -> if (timeline.startNs >= 0 && operation == Operation.STARTING) {
                            operation = Operation.RECORDING
                        }
                        is VideoRecordEvent.Finalize -> {
                            recording = null
                            bound.timeline.set(null)
                            operation = Operation.SAVING
                            val error = recordingFailure ?: if (event.hasError()) {
                                failure("Gravação encerrada (código ${event.error})", event.cause)
                            } else null
                            finishDrive(activeLog, snapshot, event.recordingStats.recordedDurationNanos / 1e9, error)
                        }
                    }
                }
        } catch (error: Exception) {
            bound.timeline.set(null)
            val message = failure("Não foi possível iniciar a gravação", error)
            if (log != null) {
                operation = Operation.SAVING
                finishDrive(log, snapshot, 0.0, message)
            } else {
                operation = Operation.IDLE
                sessionMessage = message
            }
        }
    }

    fun stopDrive() {
        operation = Operation.SAVING
        recording?.stop()
        speaker.stop()
    }

    private fun finishDrive(log: DriveLog, snapshot: Settings, duration: Double, error: String?) {
        worker.execute {
            drive = null
            var message = error
            try {
                log.drain()
            } catch (exception: Exception) {
                message = failure("Falha ao salvar as detecções", exception)
            }
            message = closeSession(log.session, duration, message)
            resetPipeline(snapshot)
            main.post {
                operation = Operation.IDLE
                if (closed) shutdownWorker()
                else {
                    sessionMessage = message ?: "Sessão salva em ${log.session.directory.name}."
                    if (camera == null) bindCamera()
                }
            }
        }
    }

    fun openReplay(uri: Uri) {
        if (!modelReady || operation != Operation.IDLE) return
        replay = true
        unbindCamera()
        clearOverlay()
        sessionMessage = null
        val snapshot = settings
        val cancelled = AtomicBoolean()
        replayCancel = cancelled
        operation = Operation.REPLAYING
        worker.execute {
            var session: SessionRecorder? = null
            var duration = 0.0
            var error: String? = null
            try {
                val file = SessionRecorder.create(context, metadata("replay", snapshot))
                session = file
                val source = VideoFrameSource(context, uri)
                source.saveVideo(file.video, cancelled::get)
                val processor = Pipeline(checkNotNull(runner), snapshot.band)
                val translator = Translator.load(context.assets, snapshot.language)
                val start = System.nanoTime()
                duration = source.run(file.video, cancelled::get) { timestamp, pixels ->
                    val result = processor.analyse(pixels, timestamp / 1e9)
                    val thumb = pixels.thumbnail()
                    val wait = (start + timestamp - System.nanoTime()) / 1_000_000
                    if (wait > 0) Thread.sleep(wait.coerceAtMost(70))
                    if (!cancelled.get()) {
                        file.frame(result.time, result.width, result.height, result.detections, result.alerts, translator)
                        val text = result.alerts.lastOrNull()?.let(translator::alert)
                        main.post {
                            if (replayCancel === cancelled && !cancelled.get()) {
                                videoFrame = thumb
                                present(result, text, speak = true)
                            }
                        }
                    }
                }
                if (cancelled.get()) throw CancellationException()
            } catch (_: CancellationException) {
                error = "Reprodução interrompida."
            } catch (exception: Exception) {
                error = failure("Falha na reprodução", exception)
            }
            val message = session?.let { closeSession(it, duration, error) } ?: error
            main.post {
                replayCancel = null
                operation = Operation.IDLE
                if (closed) shutdownWorker() else sessionMessage = message ?: "Reprodução salva em ${session?.directory?.name}."
            }
        }
    }

    fun stopReplay() {
        replayCancel?.set(true)
        speaker.stop()
        operation = Operation.SAVING
    }

    fun returnToDrive() {
        if (operation != Operation.IDLE) return
        replay = false
        videoFrame = null
        sessionMessage = null
        clearOverlay()
        bindCamera()
    }

    private fun present(result: Analysis, alert: String?, speak: Boolean) {
        frame = result
        if (alert != null) {
            lastAlert = alert
            val generation = ++alertGeneration
            main.postDelayed({ if (generation == alertGeneration) lastAlert = null }, 3_000)
            if (speak && settings.speech) speaker.speak(alert, settings.rate)
        }
    }

    private fun clearOverlay() {
        frame = null
        lastAlert = null
        alertGeneration++
    }

    private fun metadata(mode: String, config: Settings) = JSONObject().put("mode", mode)
        .put("detection_band", config.band).put("language", config.language)
        .put("exposure_ev", if (mode == "drive") config.exposure else JSONObject.NULL)
        .put("meter_on_lights", if (mode == "drive") config.meter else JSONObject.NULL)

    private fun closeSession(session: SessionRecorder, duration: Double, error: String?): String? {
        return runCatching { session.close(runner, duration, error) }
            .exceptionOrNull()?.let { failure("Falha ao concluir a sessão", it) } ?: error
    }

    private fun failure(prefix: String, error: Throwable?): String {
        Log.w("Farol", prefix, error)
        return listOfNotNull(prefix, error?.message?.takeIf { it.isNotBlank() }).joinToString(": ")
    }

    fun close() {
        closed = true
        setForeground(false)
        speaker.close()
        if (recording == null && operation != Operation.SAVING) shutdownWorker()
    }

    private fun shutdownWorker() {
        if (worker.isShutdown) return
        worker.execute { runner?.close() }
        worker.shutdown()
    }

    private class DriveLog(
        val session: SessionRecorder,
        val timeline: VideoTimeline,
        private val translator: Translator,
    ) {
        private val pending = ArrayDeque<Pair<Long, Analysis>>()

        fun enqueue(timestamp: Long, frame: Analysis) {
            check(pending.size < 120) { "O codificador de vídeo deixou de acompanhar a análise." }
            pending.addLast(timestamp to frame)
        }

        fun drain() {
            val end = timeline.endNs
            while (pending.isNotEmpty() && pending.first().first <= end + 1_000) {
                val (_, frame) = pending.removeFirst()
                session.frame(frame.time, frame.width, frame.height, frame.detections, frame.alerts, translator)
            }
        }
    }
}
