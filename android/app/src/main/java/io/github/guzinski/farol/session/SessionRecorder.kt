package io.github.guzinski.farol.session

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.os.BatteryManager
import android.os.PowerManager
import android.os.Build
import io.github.guzinski.farol.BuildConfig
import io.github.guzinski.farol.detect.Analysis
import io.github.guzinski.farol.detect.Detection
import io.github.guzinski.farol.detect.ModelRunner
import io.github.guzinski.farol.i18n.Translator
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SessionRecorder(val directory: File, private val metadata: JSONObject, private val thermal: Thermal? = null) {
    /** Reads the device temperature. getThermalHeadroom answers at most about once per second. */
    class Thermal(private val context: Context) {
        private val power = context.getSystemService(PowerManager::class.java)
        val status: Int get() = power.currentThermalStatus
        val headroom: Float get() = power.getThermalHeadroom(0)
        val batteryCelsius: Double?
            get() = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                ?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
                ?.takeIf { it != Int.MIN_VALUE }?.let { it / 10.0 }
    }

    val video = File(directory, "video.mp4")
    private val detections = File(directory, "detections.jsonl").bufferedWriter()
    private val alerts = JSONArray()
    private var first = 0.0
    private var last = 0.0
    private var firstProcessed = 0L
    private var lastProcessed = 0L
    private var frames = 0
    private var totalPreprocessMs = 0.0
    private var totalInferenceMs = 0.0
    private var totalFrameMs = 0.0
    private var nextThermal = 0L
    private var maxThermalStatus = -1
    private var minHeadroom = Float.NaN
    private var startBattery: Double? = null
    private var lastBattery: Double? = null
    private var nextCrops = Double.NEGATIVE_INFINITY
    private var savedCrops = 0

    fun frame(result: Analysis, translator: Translator) {
        lastProcessed = System.nanoTime()
        if (frames == 0) {
            first = result.time
            firstProcessed = lastProcessed
            metadata.put("analysis_resolution", JSONArray(listOf(result.width, result.height)))
        }
        last = result.time
        frames++
        totalPreprocessMs += result.preprocessMs
        totalInferenceMs += result.inferenceMs
        totalFrameMs += result.frameMs
        val line = JSONObject().put("t", result.time)
            .put("preprocess_ms", result.preprocessMs).put("inference_ms", result.inferenceMs).put("frame_ms", result.frameMs)
        if (thermal != null && lastProcessed >= nextThermal) {
            nextThermal = lastProcessed + THERMAL_INTERVAL_NS
            val status = thermal.status
            val headroom = thermal.headroom
            val battery = thermal.batteryCelsius
            maxThermalStatus = maxOf(maxThermalStatus, status)
            if (!headroom.isNaN()) minHeadroom = if (minHeadroom.isNaN()) headroom else minOf(minHeadroom, headroom)
            if (startBattery == null) startBattery = battery
            lastBattery = battery ?: lastBattery
            line.put("thermal_status", status).put("battery_c", battery ?: JSONObject.NULL)
                .put("thermal_headroom", headroom.takeUnless { it.isNaN() }?.toDouble() ?: JSONObject.NULL)
        }
        saveCrops(result)
        detections.write(line.put("detections", JSONArray(result.detections.map(::record))).toString())
        detections.newLine()
        for (alert in result.alerts) {
            val params = alert.lights?.let { lights ->
                JSONObject().put("lights", JSONArray(lights.map { JSONArray(listOf(it.position.key, it.color.key)) }))
            }
            alerts.put(JSONObject().put("t", alert.t).put("level", alert.level.key).put("key", alert.key)
                .put("params", params ?: JSONObject.NULL).put("text", translator.alert(alert)))
        }
        if (frames % 15 == 0) detections.flush()
    }

    fun close(model: ModelRunner?, elapsedSeconds: Double, error: String? = null) {
        try {
            detections.close()
        } finally {
            File(directory, "alerts.json").writeText(alerts.toString(2))
            metadata.put("device_model", Build.MODEL)
                .put("app_version", BuildConfig.GIT_COMMIT)
                .put("model_file", model?.name ?: JSONObject.NULL)
                .put("model_input_size", model?.let { JSONArray(listOf(it.width, it.height)) } ?: JSONObject.NULL)
                .put("backend", model?.backend ?: JSONObject.NULL)
                .put("analysis_fps", if (lastProcessed > firstProcessed) (frames - 1) * 1e9 / (lastProcessed - firstProcessed) else 0)
                .put("sampled_video_fps", if (last > first) (frames - 1) / (last - first) else 0)
                .put("analysed_frames", frames)
                .put("mean_preprocess_ms", if (frames > 0) totalPreprocessMs / frames else JSONObject.NULL)
                .put("mean_inference_ms", if (frames > 0) totalInferenceMs / frames else JSONObject.NULL)
                .put("mean_frame_ms", if (frames > 0) totalFrameMs / frames else JSONObject.NULL)
                .put("max_thermal_status", if (maxThermalStatus >= 0) maxThermalStatus else JSONObject.NULL)
                .put("min_thermal_headroom", minHeadroom.takeUnless { it.isNaN() }?.toDouble() ?: JSONObject.NULL)
                .put("battery_c_start", startBattery ?: JSONObject.NULL)
                .put("battery_c_end", lastBattery ?: JSONObject.NULL)
                .put("video_duration_s", elapsedSeconds)
                .put("position_basis", "image x-coordinate; the left of two side-by-side lights ahead counts as left")
                .put("uncolored_crops", savedCrops)
                .put("error", error ?: JSONObject.NULL)
            File(directory, "session.json").writeText(metadata.toString(2))
        }
    }

    // Crops of lights left without colour, at the analysis resolution, to tune the colour reader
    // on what the app actually saw. Rate-limited so a long drive stays small.
    private fun saveCrops(result: Analysis) {
        if (result.uncolored.isEmpty() || result.time < nextCrops || savedCrops >= MAX_CROPS) return
        nextCrops = result.time + CROP_INTERVAL_S
        val folder = File(directory, "crops")
        if (!folder.isDirectory && !folder.mkdir()) return
        for (crop in result.uncolored) {
            if (savedCrops >= MAX_CROPS) break
            val pixels = IntArray(crop.pixels.size) { crop.pixels[it] or (0xff shl 24) }
            val bitmap = Bitmap.createBitmap(pixels, crop.width, crop.height, Bitmap.Config.ARGB_8888)
            val name = String.format(Locale.ROOT, "%08.2f_%d_%d_%d_%d.png", result.time,
                crop.box.left, crop.box.top, crop.box.right, crop.box.bottom)
            File(folder, name).outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            bitmap.recycle()
            savedCrops++
        }
    }

    private fun record(detection: Detection): JSONObject {
        val box = detection.box
        val read = detection.light
        val counts = read?.counts?.let {
            JSONObject().put("red", it.red).put("yellow", it.yellow).put("green", it.green).put("area", it.area)
        }
        return JSONObject().put("class", detection.cls).put("conf", detection.confidence.toDouble())
            .put("track_id", detection.id).put("box", JSONArray(listOf(box.left, box.top, box.right, box.bottom)))
            .put("color", detection.color?.key ?: JSONObject.NULL)
            .put("position", detection.position?.key ?: JSONObject.NULL)
            .put("lit_pixel_counts", counts ?: JSONObject.NULL)
            .put("thirds_brightness", read?.thirds?.let { JSONArray(it) } ?: JSONObject.NULL)
            .put("seen_from_behind", read?.seenFromBehind ?: JSONObject.NULL)
            .put("lamp_by_position", read?.lampByPosition?.key ?: JSONObject.NULL)
            .put("red_fraction", detection.redFraction ?: JSONObject.NULL)
    }

    companion object {
        private const val THERMAL_INTERVAL_NS = 2_000_000_000L
        private const val CROP_INTERVAL_S = .5
        private const val MAX_CROPS = 400

        internal fun root(context: Context): File? =
            context.externalMediaDirs.firstOrNull()?.let { File(it, "sessions") }

        fun create(context: Context, metadata: JSONObject): SessionRecorder {
            val root = checkNotNull(root(context)) { "Android/media is unavailable." }
            check(root.isDirectory || root.mkdirs()) { "Could not create the sessions folder." }
            val format = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.ROOT)
            var time = System.currentTimeMillis()
            while (true) {
                val directory = File(root, format.format(Date(time)))
                if (directory.mkdir()) return SessionRecorder(directory, metadata, Thermal(context.applicationContext))
                check(directory.isDirectory) { "Could not create a new session." }
                time += 1_000
            }
        }
    }
}
