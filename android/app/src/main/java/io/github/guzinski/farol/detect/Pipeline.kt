package io.github.guzinski.farol.detect

import io.github.guzinski.farol.camera.crop
import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.perception.LightColor
import io.github.guzinski.farol.perception.Pixels
import io.github.guzinski.farol.rules.Alert
import io.github.guzinski.farol.rules.Evaluator

data class Analysis(
    val width: Int,
    val height: Int,
    val time: Double,
    val detections: List<Detection>,
    val lights: List<LampColor?>,
    val alerts: List<Alert>,
    val fps: Double,
    val preprocessMs: Double,
    val inferenceMs: Double,
    val frameMs: Double = preprocessMs + inferenceMs,
    val uncolored: List<LightCrop> = emptyList(),
)

/** Packed RGB pixels of a traffic light the colour reader could not classify, kept for review. */
class LightCrop(val box: Box, val width: Int, val height: Int, val pixels: IntArray)

class FrameGate {
    private var next = Long.MIN_VALUE
    private var previous = Long.MIN_VALUE

    fun accept(timestamp: Long): Boolean {
        if (timestamp <= previous || timestamp + 1_000_000 < next) return false
        previous = timestamp
        next = if (next == Long.MIN_VALUE) timestamp + ANALYSIS_INTERVAL_NS
        else maxOf(next + ANALYSIS_INTERVAL_NS, timestamp + 32_000_000)
        return true
    }
}

class Pipeline(private val runner: ModelRunner, private val band: Double) {
    private val tracker = IouTracker()
    private val evaluator = Evaluator()
    private val recent = ArrayDeque<Long>()

    fun analyse(frame: Pixels, t: Double): Analysis {
        val start = System.nanoTime()
        val detections = tracker.update(runner.detect(frame, band))
        for (detection in detections) {
            val crop = frame.crop(detection.box)
            if (detection.cls == TRAFFIC_LIGHT) detection.light = LightColor.read(crop)
            else if (detection.cls == STOP_SIGN) detection.redFraction = LightColor.redFraction(crop)
        }
        val uncolored = detections.asSequence()
            .filter { it.cls == TRAFFIC_LIGHT && it.color == null && it.box.height >= 8 && it.box.bottom < band * frame.height }
            .take(MAX_CROPS_PER_FRAME)
            .map { detection ->
                val crop = frame.crop(detection.box)
                LightCrop(detection.box, crop.width, crop.height, IntArray(crop.width * crop.height) {
                    crop.rgb(it % crop.width, it / crop.width)
                })
            }.toList()
        val emitted = evaluator.evaluate(t, detections, frame.width, frame.height, band, tracker.activeIds)
        recent.addLast(System.nanoTime())
        if (recent.size > 30) recent.removeFirst()
        val fps = if (recent.size > 1) (recent.size - 1) * 1e9 / (recent.last() - recent.first()) else 0.0
        return Analysis(frame.width, frame.height, t, detections, evaluator.display, emitted, fps,
            runner.preprocessMs, runner.inferenceMs, (System.nanoTime() - start) / 1e6, uncolored)
    }

    private companion object {
        const val MAX_CROPS_PER_FRAME = 4
    }
}
