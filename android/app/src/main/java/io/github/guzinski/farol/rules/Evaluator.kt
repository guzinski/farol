package io.github.guzinski.farol.rules

import io.github.guzinski.farol.detect.Detection
import io.github.guzinski.farol.detect.STOP_SIGN
import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.perception.LightState
import io.github.guzinski.farol.perception.PositionedLight
import io.github.guzinski.farol.perception.TrafficLightsByPosition
import java.math.BigDecimal
import java.math.RoundingMode

enum class Level(val key: String) { DANGER("danger"), WARNING("warning"), INFO("info") }

data class Alert(val t: Double, val level: Level, val key: String, val lights: List<PositionedLight>? = null)

class AlertManager {
    private val lastBySlot = mutableMapOf<String, Double>()
    private var last = Double.NEGATIVE_INFINITY

    fun tryEmit(
        t: Double, slot: String, level: Level, key: String, repeatAfter: Double,
        lights: List<PositionedLight>? = null,
    ): Alert? {
        if (t - lastBySlot.getOrDefault(slot, Double.NEGATIVE_INFINITY) < repeatAfter) return null
        if (t - last < if (level == Level.DANGER) .9 else 1.8) return null
        lastBySlot[slot] = t
        last = t
        return Alert(BigDecimal(t).setScale(2, RoundingMode.HALF_EVEN).toDouble(), level, key, lights)
    }
}

class Evaluator {
    private val lights = TrafficLightsByPosition()
    private var state: LightState? = null
    private val alerts = AlertManager()
    private val stopSeen = mutableMapOf<Int, Int>()
    val display get() = lights.display

    fun evaluate(
        t: Double, detections: List<Detection>, width: Int, height: Int, band: Double, activeIds: Set<Int>,
    ): List<Alert> = buildList {
        val now = lights.update(t, detections, width, height, band)
        val before = state
        state = now
        when {
            now is LightState.Mixed && now != before -> {
                val slot = now.lights.joinToString(prefix = "light-", separator = ",") { "${it.position.key}=${it.color.key}" }
                alerts.tryEmit(t, slot, Level.WARNING, "traffic_light.by_position", 3.0, now.lights)?.let { add(it) }
            }
            now is LightState.Single && now != before -> {
                val color = now.color
                val green = color == LampColor.GREEN
                val key = if (before == LightState.Single(LampColor.RED) && green) "turned_green" else color.key
                alerts.tryEmit(t, "light-${color.key}", if (green) Level.INFO else Level.WARNING, "traffic_light.$key", 3.0)
                    ?.let { add(it) }
            }
            now == LightState.Single(LampColor.RED) ->
                alerts.tryEmit(t, "light-red", Level.WARNING, "traffic_light.red", 30.0)?.let { add(it) }
        }

        stopSeen.keys.retainAll(activeIds)
        for (detection in detections) {
            if (detection.cls == STOP_SIGN && detection.confidence > .45f && (detection.redFraction ?: 0.0) > .35) {
                val count = stopSeen.getOrDefault(detection.id, 0) + 1
                stopSeen[detection.id] = count
                if (count == 4) alerts.tryEmit(t, "stop", Level.WARNING, "sign.stop", 8.0)?.let { add(it) }
            }
        }
    }
}
