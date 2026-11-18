package io.github.guzinski.farol.perception

import io.github.guzinski.farol.detect.Detection
import io.github.guzinski.farol.detect.TRAFFIC_LIGHT

const val DEFAULT_BAND = .65
enum class Position(val key: String) { LEFT("left"), AHEAD("ahead"), RIGHT("right") }
data class PositionedLight(val position: Position, val color: LampColor)

sealed interface LightState {
    data class Single(val color: LampColor) : LightState
    data class Mixed(val lights: List<PositionedLight>) : LightState
}

class TrafficLightsByPosition {
    private val votes = Position.entries.associateWith { ArrayDeque<LampColor?>() }
    private val colors = arrayOfNulls<LampColor>(Position.entries.size)
    private var previousTime: Double? = null
    val display: List<LampColor?> get() = colors.toList()

    fun update(t: Double, detections: List<Detection>, width: Int, height: Int, band: Double): LightState? {
        val gap = previousTime?.let { t - it } ?: 0.0
        previousTime = t
        val candidates = detections.filter { it.cls == TRAFFIC_LIGHT && it.color != null && it.box.bottom < band * height }
        val minHeight = (candidates.maxOfOrNull { it.box.height } ?: 0) * .5
        val best = mutableMapOf<Position, Detection>()
        for (candidate in candidates.filter { it.box.height >= minHeight }) {
            val x = (candidate.box.left + candidate.box.right) / 2.0
            val position = when {
                x < .38 * width -> Position.LEFT
                x > .62 * width -> Position.RIGHT
                else -> Position.AHEAD
            }
            candidate.position = position
            if (candidate.box.height > (best[position]?.box?.height ?: 0)) best[position] = candidate
        }
        for (position in Position.entries) {
            val window = votes.getValue(position)
            val color = best[position]?.color
            if (gap > SLOW_FRAME_GAP_SECONDS) {
                val previous = window.lastOrNull()
                // Consecutive agreeing frames can confirm a color when frames are sparse.
                // A missing reading or a long stall clears the last confirmed color.
                if (color == null || gap > MAX_CONFIRMATION_GAP_SECONDS) {
                    colors[position.ordinal] = null
                } else if (color == previous) {
                    colors[position.ordinal] = color
                }
                window.clear()
                window.addLast(color)
                continue
            }
            if (window.size == 9) window.removeFirst()
            window.addLast(color)
            if (window.size == 9) {
                val winner = window.groupingBy { it }.eachCount().entries.maxBy { it.value }
                if (winner.value >= 6) colors[position.ordinal] = winner.key
            }
        }
        val seen = Position.entries.mapNotNull { position ->
            colors[position.ordinal]?.let { PositionedLight(position, it) }
        }
        if (seen.isEmpty()) return null
        return if (seen.all { it.color == seen.first().color }) {
            LightState.Single(seen.first().color)
        } else LightState.Mixed(seen)
    }

    private companion object {
        const val SLOW_FRAME_GAP_SECONDS = .16
        const val MAX_CONFIRMATION_GAP_SECONDS = 1.5
    }
}
