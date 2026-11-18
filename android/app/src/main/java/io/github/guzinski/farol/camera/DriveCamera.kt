package io.github.guzinski.farol.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Rational
import android.util.Size
import android.view.Surface
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.SessionConfig
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.RecorderClockBridge
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.roundToInt

class VideoTimeline(private val uptimeToSensorNs: Long) {
    private val first = AtomicLong(-1)
    private val last = AtomicLong(-1)
    val startNs: Long get() = first.get()
    val endNs: Long get() = last.get()

    fun encoded(ptsUs: Long) {
        val sensorNs = ptsUs * 1_000 + uptimeToSensorNs
        first.compareAndSet(-1, sensorNs)
        last.set(sensorNs)
    }
}

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
class DriveCamera(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val provider: ProcessCameraProvider,
    private val previewView: PreviewView,
    private val worker: Executor,
) {
    data class Bound(
        val camera: Camera,
        val video: VideoCapture<Recorder>,
        val analysis: ImageAnalysis,
        val videoQuality: String,
        val lens: String,
        val realtimeTimestamps: Boolean,
        val timeline: AtomicReference<VideoTimeline?>,
    ) {
        fun newTimeline(): VideoTimeline {
            var offset = 0L
            if (realtimeTimestamps) {
                var bestGap = Long.MAX_VALUE
                repeat(5) {
                    val before = System.nanoTime()
                    val realtime = SystemClock.elapsedRealtimeNanos()
                    val after = System.nanoTime()
                    if (after - before < bestGap) {
                        bestGap = after - before
                        offset = realtime - (before + bestGap / 2)
                    }
                }
            }
            return VideoTimeline(offset).also { timeline.set(it) }
        }

        fun setExposure(ev: Float): Int {
            val exposure = camera.cameraInfo.exposureState
            val step = exposure.exposureCompensationStep.toFloat()
            if (step <= 0f) return 0
            val index = (ev / step).roundToInt().coerceIn(
                exposure.exposureCompensationRange.lower, exposure.exposureCompensationRange.upper,
            )
            camera.cameraControl.setExposureCompensationIndex(index)
            return index
        }

        fun meterOnLights(preview: PreviewView) {
            if (preview.width == 0 || preview.height == 0) return
            val ae = preview.meteringPointFactory.createPoint(.5f * preview.width, .27f * preview.height, .24f)
            camera.cameraControl.startFocusAndMetering(
                FocusMeteringAction.Builder(ae, FocusMeteringAction.FLAG_AE)
                    .setAutoCancelDuration(5, TimeUnit.SECONDS).build(),
            )
        }
    }

    fun bind(use4k: Boolean, analyzer: ImageAnalysis.Analyzer): Bound {
        val logical = provider.getCameraInfo(CameraSelector.DEFAULT_BACK_CAMERA)
        val baseId = Camera2CameraInfo.from(logical).cameraId
        val manager = context.getSystemService(CameraManager::class.java)
        val base = manager.getCameraCharacteristics(baseId)
        // Select the S25+'s approximately 23 mm equivalent rear wide lens.
        val physical = base.physicalCameraIds.minByOrNull { id ->
            val info = manager.getCameraCharacteristics(id)
            val focal = info.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
            val sensor = info.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            if (focal == null || sensor == null) Float.POSITIVE_INFINITY else abs(36 * focal / sensor.width - 23f)
        }
        val selector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).apply {
            if (physical != null) setPhysicalCameraId(physical)
            else check(base.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                ?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA) != true) {
                "CameraX did not expose the rear wide lens."
            }
        }.build()
        val realtime = base.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
            CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        val rotation = previewView.display?.rotation ?: Surface.ROTATION_90
        val viewport = ViewPort.Builder(Rational(16, 9), rotation).setScaleType(ViewPort.FILL_CENTER).build()
        val size = if (use4k) Size(3840, 2160) else Size(1920, 1080)
        val attempts = listOf(Quality.FHD to true, Quality.HD to true, Quality.HD to false)
        val fps = Range(30, 30)
        var failure: Exception? = null
        for ((quality, exact) in attempts) {
            provider.unbindAll()
            val timeline = AtomicReference<VideoTimeline?>(null)
            val recorder = RecorderClockBridge.attach(Recorder.Builder()) { timeline.get()?.encoded(it) }
                .setQualitySelector(QualitySelector.from(quality)).build()
            val video = VideoCapture.Builder(recorder).setTargetRotation(rotation).build()
            val preview = Preview.Builder().setTargetRotation(rotation).build()
            val resolver = ResolutionSelector.Builder().setResolutionStrategy(
                ResolutionStrategy(size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER),
            ).setResolutionFilter { resolutions, _ ->
                resolutions.filter {
                    if (exact) it == size else it.width <= size.width && it.height <= size.height
                }
            }.build()
            val builder = ImageAnalysis.Builder().setTargetRotation(rotation).setResolutionSelector(resolver)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            Camera2Interop.Extender(builder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            val analysis = builder.build().apply { setAnalyzer(worker, analyzer) }
            preview.surfaceProvider = previewView.surfaceProvider
            val config = SessionConfig.Builder(listOf(preview, analysis, video))
                .setViewPort(viewport).setFrameRateRange(fps).build()
            val qualityName = if (quality == Quality.FHD) "1080p" else "720p"
            try {
                val supported = runCatching { provider.getCameraInfo(selector).isSessionConfigSupported(config) }
                    .onFailure { Log.w("FarolCamera", "Session support query failed", it) }.getOrNull()
                Log.i("FarolCamera", "Camera ${physical ?: baseId}, analysis $size exact=$exact, video $qualityName, query=$supported")
                val camera = provider.bindToLifecycle(owner, selector, config)
                Log.i("FarolCamera", "Bound preview ${preview.resolutionInfo?.resolution}, analysis ${analysis.resolutionInfo?.resolution}, video ${video.resolutionInfo?.resolution}")
                return Bound(camera, video, analysis, qualityName, physical ?: baseId, realtime, timeline)
            } catch (error: IllegalArgumentException) {
                failure = error
                analysis.clearAnalyzer()
            } catch (error: IllegalStateException) {
                failure = error
                analysis.clearAnalyzer()
            }
        }
        provider.unbindAll()
        throw IllegalStateException("Could not bind the wide camera at 30 fps.", failure)
    }
}
