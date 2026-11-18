package io.github.guzinski.farol.session

import io.github.guzinski.farol.detect.STOP_SIGN
import io.github.guzinski.farol.detect.TRAFFIC_LIGHT
import io.github.guzinski.farol.perception.LampColor
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException

data class SavedSession(
    val directory: File,
    val mode: String?,
    val duration: Double?,
    val frameCount: Int?,
    val fps: Double?,
    val resolution: String?,
    val model: String?,
    val backend: String?,
    val error: String?,
    val finalized: Boolean,
) {
    val id: String get() = directory.name
    val video: File get() = File(directory, "video.mp4")
}

data class SavedAlert(val seconds: Double, val text: String)

data class DetectionSummary(
    val frames: Int,
    val lightTracks: Int,
    val stopTracks: Int,
    val lightObservations: Int,
    val colors: Map<LampColor, Int>,
    val unreadableLines: Int,
)

data class SessionDetails(
    val alerts: List<SavedAlert>?,
    val detections: DetectionSummary?,
)

object SessionHistory {
    private val sessionName = Regex("""[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}""")

    fun list(root: File?): List<SavedSession> {
        return root?.listFiles { file -> file.isDirectory && sessionName.matches(file.name) }
            .orEmpty().sortedByDescending { it.name }.map { directory ->
                val data = readFile(File(directory, "session.json")) { JSONObject(it.readText()) }
                val size = data?.optJSONArray("analysis_resolution")
                val width = size?.optInt(0)
                val height = size?.optInt(1)
                SavedSession(
                    directory = directory,
                    mode = data?.text("mode"),
                    duration = data?.optDouble("video_duration_s")?.takeIf { it.isFinite() && it >= 0 },
                    frameCount = data?.optInt("analysed_frames", -1)?.takeIf { it >= 0 },
                    fps = data?.optDouble("analysis_fps")?.takeIf { it.isFinite() && it >= 0 },
                    resolution = if (width != null && width > 0 && height != null && height > 0) "$width × $height" else null,
                    model = data?.text("model_file"),
                    backend = data?.text("backend"),
                    error = data?.text("error"),
                    finalized = data != null,
                )
            }
    }

    fun details(session: SavedSession): SessionDetails {
        val alerts = readFile(File(session.directory, "alerts.json")) { file ->
            val data = JSONArray(file.readText())
            (0 until data.length()).mapNotNull { index ->
                val entry = data.optJSONObject(index) ?: return@mapNotNull null
                val seconds = entry.optDouble("t")
                val text = entry.text("text") ?: entry.text("key")
                if (seconds.isFinite() && seconds >= 0 && text != null) SavedAlert(seconds, text) else null
            }.sortedBy { it.seconds }
        }
        val detections = readFile(File(session.directory, "detections.jsonl"), ::summarize)
        return SessionDetails(alerts, detections)
    }

    private fun summarize(file: File): DetectionSummary {
        val lightTracks = mutableSetOf<Int>()
        val stopTracks = mutableSetOf<Int>()
        val colors = mutableMapOf<LampColor, Int>()
        val palette = LampColor.entries.associateBy { it.key }
        var frames = 0
        var lightObservations = 0
        var unreadable = 0
        file.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val detections = try {
                JSONObject(line).getJSONArray("detections")
            } catch (_: org.json.JSONException) {
                unreadable++
                return@forEachLine
            }
            frames++
            for (index in 0 until detections.length()) {
                val entry = detections.optJSONObject(index) ?: continue
                val track = entry.optInt("track_id", -1)
                when (entry.optInt("class", -1)) {
                    TRAFFIC_LIGHT -> {
                        lightObservations++
                        if (track >= 0) lightTracks += track
                        palette[entry.text("color")]?.let { color -> colors[color] = colors.getOrDefault(color, 0) + 1 }
                    }
                    STOP_SIGN -> if (track >= 0) stopTracks += track
                }
            }
        }
        return DetectionSummary(frames, lightTracks.size, stopTracks.size, lightObservations, colors.toMap(), unreadable)
    }

    private fun JSONObject.text(key: String): String? = (opt(key) as? String)?.takeIf { it.isNotBlank() }

    private fun <T> readFile(file: File, read: (File) -> T): T? = try {
        if (file.isFile) read(file) else null
    } catch (_: IOException) {
        null
    } catch (_: org.json.JSONException) {
        null
    }
}
