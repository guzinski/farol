package io.github.guzinski.farol

import io.github.guzinski.farol.camera.VideoTimeline
import io.github.guzinski.farol.detect.Box
import io.github.guzinski.farol.detect.Detection
import io.github.guzinski.farol.detect.FrameGate
import io.github.guzinski.farol.detect.IouTracker
import io.github.guzinski.farol.detect.Letterbox
import io.github.guzinski.farol.detect.STOP_SIGN
import io.github.guzinski.farol.detect.TRAFFIC_LIGHT
import io.github.guzinski.farol.detect.YoloOutput
import io.github.guzinski.farol.detect.suppress
import io.github.guzinski.farol.i18n.Translator
import io.github.guzinski.farol.perception.DEFAULT_BAND
import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.perception.LightRead
import io.github.guzinski.farol.perception.LightState
import io.github.guzinski.farol.perception.Position
import io.github.guzinski.farol.perception.PositionedLight
import io.github.guzinski.farol.perception.TrafficLightsByPosition
import io.github.guzinski.farol.rules.Alert
import io.github.guzinski.farol.rules.AlertManager
import io.github.guzinski.farol.rules.Evaluator
import io.github.guzinski.farol.rules.Level
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class PipelineTest {
    @Test
    fun votesRequireSixOfNineAndRememberUntilSixNulls() {
        val votes = TrafficLightsByPosition()
        val left = light(20, LampColor.RED)
        val right = light(160, LampColor.GREEN)
        val small = light(100, LampColor.YELLOW, height = 5)
        repeat(8) { assertNull(votes.update(listOf(left, right, small), 200, 100, DEFAULT_BAND)) }
        assertEquals(LightState.Mixed(listOf(PositionedLight(Position.LEFT, LampColor.RED),
            PositionedLight(Position.RIGHT, LampColor.GREEN))),
            votes.update(listOf(left, right, small), 200, 100, DEFAULT_BAND))
        assertNull(small.position)
        repeat(5) { assertEquals(2, (votes.update(emptyList(), 200, 100, DEFAULT_BAND) as LightState.Mixed).lights.size) }
        assertNull(votes.update(emptyList(), 200, 100, DEFAULT_BAND))
    }

    @Test
    fun votesAllowThreeMissesAndRequireFullWindow() {
        val votes = TrafficLightsByPosition()
        val ahead = light(100, LampColor.YELLOW)
        repeat(3) { assertNull(votes.update(emptyList(), 200, 100, DEFAULT_BAND)) }
        repeat(5) { assertNull(votes.update(listOf(ahead), 200, 100, DEFAULT_BAND)) }
        assertEquals(LightState.Single(LampColor.YELLOW), votes.update(listOf(ahead), 200, 100, DEFAULT_BAND))
    }

    @Test
    fun alertsRespectGlobalGapPriorityAndSlotCooldown() {
        val alerts = AlertManager()
        assertNotNull(alerts.tryEmit(0.0, "red", Level.WARNING, "red", 3.0))
        assertNull(alerts.tryEmit(1.79, "stop", Level.WARNING, "stop", 8.0))
        assertNotNull(alerts.tryEmit(1.0, "danger", Level.DANGER, "danger", 4.0))
        assertNull(alerts.tryEmit(2.9, "red", Level.WARNING, "red", 3.0))
        assertNotNull(alerts.tryEmit(3.0, "red", Level.WARNING, "red", 3.0))
        assertNotNull(alerts.tryEmit(5.0, "stop", Level.WARNING, "stop", 8.0))
    }

    @Test
    fun evaluatorEmitsRedThenGreenAndOneStopAtFourthSighting() {
        val evaluator = Evaluator()
        fun analyse(t: Int, vararg ds: Detection) = evaluator.evaluate(t.toDouble(), ds.toList(), 200, 100, DEFAULT_BAND, setOf(1))
        val red = light(100, LampColor.RED)
        val green = light(100, LampColor.GREEN)
        repeat(8) { assertTrue(analyse(it, red).isEmpty()) }
        assertEquals("traffic_light.red", analyse(8, red).single().key)
        repeat(5) { assertTrue(analyse(9 + it, green).isEmpty()) }
        assertEquals("traffic_light.turned_green", analyse(14, green).single().key)
        val stop = Detection(STOP_SIGN, .70f, Box(20, 10, 40, 30), id = 1).apply { redFraction = .5 }
        repeat(3) { assertTrue(analyse(16 + it, stop).isEmpty()) }
        assertEquals("sign.stop", analyse(19, stop).single().key)
        assertTrue(analyse(30, stop).isEmpty())
    }

    @Test
    fun trackerKeepsClassAndIdsAcrossGapsWithoutSharingAnId() {
        val tracker = IouTracker()
        val light = light(50, LampColor.RED)
        val first = tracker.update(listOf(light)).single()
        repeat(3) { tracker.update(emptyList()) }
        val same = tracker.update(listOf(light, light, light.copy(cls = STOP_SIGN)))
        assertEquals(first.id, same.first().id)
        assertEquals(3, same.map { it.id }.toSet().size)
        repeat(10) { tracker.update(emptyList()) }
        assertTrue(tracker.activeIds.isEmpty())
        assertNotEquals(first.id, tracker.update(listOf(light)).single().id)
    }

    @Test
    fun nmsSuppressesOnlyOverlappingBoxesOfTheSameClass() {
        val light = light(50, LampColor.RED)
        val overlap = light(52, LampColor.RED).copy(confidence = .65f)
        val far = light(100, LampColor.RED)
        val stop = overlap.copy(cls = STOP_SIGN)
        val kept = suppress(listOf(overlap, light, far, stop), Detection::confidence, Detection::cls) { a, b ->
            a.box.iou(b.box)
        }
        assertEquals(listOf(light, far, stop), kept)
    }

    @Test
    fun decoderReadsBothNormalizedLayoutsAndRejectsAnotherWinningClass() {
        for (channelsFirst in listOf(true, false)) {
            val transform = Letterbox(200, 100, 100, 100)
            val output = YoloOutput(if (channelsFirst) intArrayOf(1, 84, 2) else intArrayOf(1, 2, 84), 100, 100)
            fun set(i: Int, c: Int, n: Float) = output.buffer.putFloat(4 * (if (channelsFirst) c * 2 + i else i * 84 + c), n)
            for (i in 0..1) {
                set(i, 0, .5f)
                set(i, 1, .5f)
                set(i, 2, .25f)
                set(i, 3, .25f)
                set(i, 4 + TRAFFIC_LIGHT, .6f)
            }
            set(1, 4, .8f)
            val detection = output.decode(200, 150, transform).single()
            assertEquals(TRAFFIC_LIGHT, detection.cls)
            assertEquals(Box(75, 25, 125, 75), detection.box)
        }
    }

    @Test
    fun frameGateSamplesThirtyFpsAtFifteen() {
        val gate = FrameGate()
        val analysed = (0 until 300).count { gate.accept(it * 1_000_000_000L / 30) }
        assertEquals(150, analysed)
    }

    @Test
    fun recordingTimelineKeepsFirstMuxedPtsInTheCameraClock() {
        val clock = VideoTimeline(uptimeToSensorNs = 500_000_000)
        assertEquals(-1L, clock.startNs)
        clock.encoded(1_000_000)
        clock.encoded(1_033_333)
        assertEquals(1_500_000_000L, clock.startNs)
        assertEquals(1_533_333_000L, clock.endNs)
    }

    @Test
    fun translatorRendersEachPositionAndFallsBackToEnglish() {
        val locales = File("src/main/assets/locales")
        val english = File(locales, "en.json").readText()
        val brazilian = File(locales, "pt-BR.json").readText()
        val alert = Alert(0.0, Level.WARNING, "traffic_light.by_position", listOf(
            PositionedLight(Position.LEFT, LampColor.RED), PositionedLight(Position.AHEAD, LampColor.GREEN),
        ))
        assertEquals("Semáforos: esquerda vermelho, em frente verde", Translator(english, brazilian).alert(alert))
        assertEquals("Traffic lights: left red, ahead green", Translator(english, "{}").alert(alert))
    }

    private fun light(x: Int, color: LampColor, height: Int = 20) =
        Detection(TRAFFIC_LIGHT, .8f, Box(x - 5, 10, x + 5, 10 + height)).apply {
            light = LightRead(null, null, null, false, color)
        }
}
