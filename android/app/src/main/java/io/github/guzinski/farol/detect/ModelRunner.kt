package io.github.guzinski.farol.detect

import io.github.guzinski.farol.perception.Pixels
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.roundToInt

interface ModelRunner : Closeable {
    val name: String
    val backend: String
    val width: Int
    val height: Int
    val preprocessMs: Double
    val inferenceMs: Double
    fun detect(frame: Pixels, band: Double): List<Detection>
}

internal class Letterbox(frameWidth: Int, val bandHeight: Int, modelWidth: Int, modelHeight: Int) {
    val scale = minOf(modelWidth.toDouble() / frameWidth, modelHeight.toDouble() / bandHeight)
    val width = (frameWidth * scale).roundToInt().coerceAtLeast(1)
    val height = (bandHeight * scale).roundToInt().coerceAtLeast(1)
    val left = (modelWidth - width) / 2
    val top = (modelHeight - height) / 2
    private val scaleX = width.toDouble() / frameWidth
    private val scaleY = height.toDouble() / bandHeight

    fun box(raw: ModelBox, frameWidth: Int, frameHeight: Int): Box? {
        fun x(coord: Float) = ((coord - left) / scaleX).toInt().coerceIn(0, frameWidth)
        fun y(coord: Float) = ((coord - top) / scaleY).toInt().coerceIn(0, frameHeight)
        return Box(x(raw.left), y(raw.top), x(raw.right), y(raw.bottom))
            .takeIf { it.width > 0 && it.height > 0 }
    }
}

internal class YoloInput(private val width: Int, private val height: Int) {
    private val buffer = ByteBuffer.allocateDirect(width * height * 3 * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
    private val floats = buffer.asFloatBuffer()
    private val pixels = FloatArray(width * height * 3)
    private val normalized = FloatArray(256) { it / 255f }

    private class Axis(source: Int, target: Int) {
        private val positions = DoubleArray(target) { ((it + .5) * source / target - .5).coerceIn(0.0, source - 1.0) }
        val lower = IntArray(target) { floor(positions[it]).toInt() }
        val upper = IntArray(target) { (lower[it] + 1).coerceAtMost(source - 1) }
        val fraction = FloatArray(target) { (positions[it] - lower[it]).toFloat() }
    }

    private data class Size(val frameWidth: Int, val bandHeight: Int, val width: Int, val height: Int)
    private var axes: Triple<Size, Axis, Axis>? = null
    private val row = IntArray(width)

    fun prepare(frame: Pixels, transform: Letterbox): ByteBuffer {
        val size = Size(frame.width, transform.bandHeight, transform.width, transform.height)
        val (_, xAxis, yAxis) = axes?.takeIf { it.first == size }
            ?: Triple(size, Axis(frame.width, transform.width), Axis(transform.bandHeight, transform.height))
                .also { axes = it }
        pixels.fill(normalized[114])
        for (line in 0 until transform.height) {
            frame.bilinearRow(
                yAxis.lower[line], yAxis.upper[line], yAxis.fraction[line],
                xAxis.lower, xAxis.upper, xAxis.fraction, transform.width, row,
            )
            var offset = ((line + transform.top) * width + transform.left) * 3
            for (col in 0 until transform.width) {
                val rgb = row[col]
                pixels[offset++] = normalized[(rgb shr 16) and 255]
                pixels[offset++] = normalized[(rgb shr 8) and 255]
                pixels[offset++] = normalized[rgb and 255]
            }
        }
        floats.clear()
        floats.put(pixels)
        buffer.rewind()
        return buffer
    }
}

internal data class ModelBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    fun iou(other: ModelBox): Double {
        val intersection = (minOf(right, other.right) - maxOf(left, other.left)).coerceAtLeast(0f) *
            (minOf(bottom, other.bottom) - maxOf(top, other.top)).coerceAtLeast(0f)
        val union = (right - left) * (bottom - top) + (other.right - other.left) * (other.bottom - other.top) - intersection
        return if (union > 0) intersection.toDouble() / union else 0.0
    }
}

internal class YoloOutput(shape: IntArray, private val width: Int, private val height: Int) {
    private data class Candidate(val cls: Int, val confidence: Float, val box: ModelBox)
    private val channels = 84
    private val channelsFirst = shape.size == 3 && shape[1] == channels
    private val candidates: Int

    init {
        require(shape.size == 3 && shape[0] == 1 && (channelsFirst || shape[2] == channels)) {
            "Expected raw COCO YOLO output [1,84,N] or [1,N,84], got ${shape.contentToString()}"
        }
        candidates = if (channelsFirst) shape[2] else shape[1]
        require(candidates > 0)
    }

    val buffer: ByteBuffer = ByteBuffer.allocateDirect(channels * candidates * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
    private val floats = buffer.asFloatBuffer()

    private fun value(channel: Int, index: Int) =
        floats.get(if (channelsFirst) channel * candidates + index else index * channels + channel)

    fun decode(frameWidth: Int, frameHeight: Int, transform: Letterbox): List<Detection> {
        val detections = mutableListOf<Candidate>()
        for (i in 0 until candidates) {
            val light = value(4 + TRAFFIC_LIGHT, i)
            val stop = value(4 + STOP_SIGN, i)
            val cls = if (light >= stop) TRAFFIC_LIGHT else STOP_SIGN
            val confidence = maxOf(light, stop)
            if (!confidence.isFinite() || confidence < .30f) continue
            // Ultralytics uses one winning class per anchor before filtering by class.
            if ((0 until 80).any { it != cls && (value(4 + it, i) > confidence || (it < cls && value(4 + it, i) == confidence)) }) continue
            val cx = value(0, i) * width
            val cy = value(1, i) * height
            val w = value(2, i) * width
            val h = value(3, i) * height
            if (!cx.isFinite() || !cy.isFinite() || !w.isFinite() || !h.isFinite() || w <= 0 || h <= 0) continue
            detections += Candidate(cls, confidence, ModelBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        }
        return suppress(detections, Candidate::confidence, Candidate::cls) { a, b -> a.box.iou(b.box) }
            .mapNotNull { candidate ->
                transform.box(candidate.box, frameWidth, frameHeight)?.let { Detection(candidate.cls, candidate.confidence, it) }
            }
    }
}
