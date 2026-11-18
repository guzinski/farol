package io.github.guzinski.farol.session

import android.content.Context
import android.os.Build
import io.github.guzinski.farol.BuildConfig
import io.github.guzinski.farol.detect.Detection
import io.github.guzinski.farol.detect.ModelRunner
import io.github.guzinski.farol.i18n.Translator
import io.github.guzinski.farol.rules.Alert
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SessionRecorder(val directory: File, private val metadata: JSONObject) {
    val video = File(directory, "video.mp4")
    private val detections = File(directory, "detections.jsonl").bufferedWriter()
    private val alerts = JSONArray()
    private var first = 0.0
    private var last = 0.0
    private var firstProcessed = 0L
    private var lastProcessed = 0L
    private var frames = 0

    fun frame(time: Double, width: Int, height: Int, results: List<Detection>, emitted: List<Alert>, translator: Translator) {
        lastProcessed = System.nanoTime()
        if (frames == 0) {
            first = time
            firstProcessed = lastProcessed
            metadata.put("analysis_resolution", JSONArray(listOf(width, height)))
        }
        last = time
        frames++
        detections.write(JSONObject().put("t", time).put("detections", JSONArray(results.map(::record))).toString())
        detections.newLine()
        for (alert in emitted) {
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
                .put("video_duration_s", elapsedSeconds)
                .put("position_basis", "image x-coordinate")
                .put("error", error ?: JSONObject.NULL)
            File(directory, "session.json").writeText(metadata.toString(2))
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
        fun create(context: Context, metadata: JSONObject): SessionRecorder {
            val media = checkNotNull(context.externalMediaDirs.firstOrNull()) { "Android/media is unavailable." }
            val root = File(media, "sessions")
            check(root.isDirectory || root.mkdirs()) { "Could not create the sessions folder." }
            val format = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.ROOT)
            var time = System.currentTimeMillis()
            while (true) {
                val directory = File(root, format.format(Date(time)))
                if (directory.mkdir()) return SessionRecorder(directory, metadata)
                check(directory.isDirectory) { "Could not create a new session." }
                time += 1_000
            }
        }
    }
}
