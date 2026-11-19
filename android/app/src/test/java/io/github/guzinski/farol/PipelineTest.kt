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
import io.github.guzinski.farol.detect.YoloInput
import io.github.guzinski.farol.detect.suppress
import io.github.guzinski.farol.i18n.Translator
import io.github.guzinski.farol.perception.DEFAULT_BAND
import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.perception.LightRead
import io.github.guzinski.farol.perception.LightState
import io.github.guzinski.farol.perception.Pixels
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
        var frame = 0
        fun feed(vararg detections: Detection) = votes.update(frame++ / 15.0, detections.toList(), 200, 100, DEFAULT_BAND)
        val left = light(20, LampColor.RED)
        val right = light(160, LampColor.GREEN)
        val small = light(100, LampColor.YELLOW, height = 5)
        repeat(8) { assertNull(feed(left, right, small)) }
        assertEquals(LightState.Mixed(listOf(PositionedLight(Position.LEFT, LampColor.RED),
            PositionedLight(Position.RIGHT, LampColor.GREEN))),
            feed(left, right, small))
        assertNull(small.position)
        repeat(5) { assertEquals(2, (feed() as LightState.Mixed).lights.size) }
        assertNull(feed())
    }

    @Test
    fun votesAllowThreeMissesAndRequireFullWindow() {
        val votes = TrafficLightsByPosition()
        var frame = 0
        fun feed(vararg detections: Detection) = votes.update(frame++ / 15.0, detections.toList(), 200, 100, DEFAULT_BAND)
        val ahead = light(100, LampColor.YELLOW)
        repeat(3) { assertNull(feed()) }
        repeat(5) { assertNull(feed(ahead)) }
        assertEquals(LightState.Single(LampColor.YELLOW), feed(ahead))
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
        var frame = 0
        fun analyse(vararg ds: Detection) = evaluator.evaluate(frame++ / 15.0, ds.toList(), 200, 100, DEFAULT_BAND, setOf(1))
        val red = light(100, LampColor.RED)
        val green = light(100, LampColor.GREEN)
        repeat(8) { assertTrue(analyse(red).isEmpty()) }
        assertEquals("traffic_light.red", analyse(red).single().key)
        repeat(40) { assertTrue(analyse(red).isEmpty()) }
        repeat(5) { assertTrue(analyse(green).isEmpty()) }
        assertEquals("traffic_light.turned_green", analyse(green).single().key)
        repeat(30) { assertTrue(analyse().isEmpty()) }
        val stop = Detection(STOP_SIGN, .70f, Box(20, 10, 40, 30), id = 1).apply { redFraction = .5 }
        repeat(3) { assertTrue(analyse(stop).isEmpty()) }
        assertEquals("sign.stop", analyse(stop).single().key)
        assertTrue(analyse(stop).isEmpty())
    }

    @Test
    fun slowFramesRequireTwoMatchingReadsAndClearAfterStalls() {
        val votes = TrafficLightsByPosition()
        val green = light(100, LampColor.GREEN)
        val red = light(100, LampColor.RED)
        fun feed(t: Double, vararg ds: Detection) = votes.update(t, ds.toList(), 200, 100, DEFAULT_BAND)
        assertNull(feed(57.22))
        assertNull(feed(58.35, green))
        assertEquals(LightState.Single(LampColor.GREEN), feed(59.49, green))
        assertEquals(LightState.Single(LampColor.GREEN), feed(60.62))
        assertNull(feed(61.75))
        assertNull(feed(76.52, green))
        assertNull(feed(77.65, red))
        assertNull(feed(80.0, red))
        assertEquals(LightState.Single(LampColor.RED), feed(81.1, red))
    }

    @Test
    fun slowFramesKeepAColorThroughOneMissedReading() {
        val evaluator = Evaluator()
        val red = light(100, LampColor.RED)
        fun analyse(t: Double, vararg ds: Detection) = evaluator.evaluate(t, ds.toList(), 200, 100, DEFAULT_BAND, emptySet())
        // Session 2026-11-19_11-39-43: a bump loses the light for one frame at 2.5 fps.
        assertTrue(analyse(47.15, red).isEmpty())
        assertEquals("traffic_light.red", analyse(47.55, red).single().key)
        assertTrue(analyse(48.02, red).isEmpty())
        assertTrue(analyse(48.42).isEmpty())
        assertTrue(analyse(48.85, red).isEmpty())
        assertTrue(analyse(49.29).isEmpty())
        assertTrue(analyse(49.65, red).isEmpty())
        assertTrue(analyse(50.05, red).isEmpty())
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
    fun modelInputResetsPaddingWhenReusingItsBuffer() {
        val input = YoloInput(2, 2)
        val frame = object : Pixels {
            override val width = 2
            override val height = 2
            override fun rgb(x: Int, y: Int) = 0xff8000
        }
        input.prepare(frame, Letterbox(2, 2, 2, 2))
        val next = input.prepare(frame, Letterbox(2, 1, 2, 2)).asFloatBuffer()
        assertArrayEquals(
            floatArrayOf(1f, 128 / 255f, 0f, 1f, 128 / 255f, 0f) + FloatArray(6) { 114 / 255f },
            FloatArray(12) { next.get(it) },
            0f,
        )
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
