package io.github.guzinski.farol.detect

import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.perception.LightRead
import io.github.guzinski.farol.perception.Position

const val TRAFFIC_LIGHT = 9
const val STOP_SIGN = 11
const val ANALYSIS_INTERVAL_NS = 1_000_000_000L / 15

data class Box(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width get() = right - left
    val height get() = bottom - top
    fun iou(other: Box): Double {
        val intersection = (minOf(right, other.right) - maxOf(left, other.left)).coerceAtLeast(0).toDouble() *
            (minOf(bottom, other.bottom) - maxOf(top, other.top)).coerceAtLeast(0)
        val union = width.toDouble() * height + other.width.toDouble() * other.height - intersection
        return if (union > 0) intersection / union else 0.0
    }
}

data class Detection(val cls: Int, val confidence: Float, val box: Box, val id: Int = 0) {
    var light: LightRead? = null
    var position: Position? = null
    var redFraction: Double? = null
    val color: LampColor? get() = light?.color
}

internal fun <T> suppress(
    detections: List<T>, confidence: (T) -> Float, cls: (T) -> Int, overlap: (T, T) -> Double,
): List<T> {
    val kept = mutableListOf<T>()
    for (detection in detections.sortedByDescending(confidence)) {
        if (kept.none { cls(it) == cls(detection) && overlap(it, detection) > .45 }) {
            kept += detection
        }
    }
    return kept
}

class IouTracker {
    private data class Track(val detection: Detection, val frame: Long)
    private val tracks = mutableMapOf<Int, Track>()
    private var frame = 0L
    private var nextId = 1
    val activeIds: Set<Int> get() = tracks.keys

    fun update(detections: List<Detection>): List<Detection> {
        frame++
        val unmatched = tracks.keys.toMutableSet()
        val result = detections.map { detection ->
            val match = unmatched.asSequence()
                .map { tracks.getValue(it).detection }
                .filter { it.cls == detection.cls }
                .map { it.id to it.box.iou(detection.box) }
                .filter { it.second >= .3 }
                .maxByOrNull { it.second }?.first
            val id = match ?: nextId++
            unmatched.remove(id)
            detection.copy(id = id).also { tracks[id] = Track(it, frame) }
        }
        tracks.values.removeAll { frame - it.frame >= 10 }
        return result
    }
}
