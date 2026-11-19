package io.github.guzinski.farol

import io.github.guzinski.farol.camera.YuvFrame
import io.github.guzinski.farol.camera.YuvMatrix
import io.github.guzinski.farol.detect.Letterbox
import io.github.guzinski.farol.detect.YoloInput
import io.github.guzinski.farol.perception.Pixels
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.random.Random

class YuvFrameTest {
    private fun frame(width: Int, height: Int, tenBit: Boolean, seed: Int, contrast: Int = 1): YuvFrame {
        val random = Random(seed)
        val bytes = if (tenBit) 2 else 1
        // Smooth gradients with noise, like a camera frame, so clamping rarely triggers.
        fun plane(w: Int, h: Int, pixel: Int, base: (Int, Int) -> Int): YuvFrame.Plane {
            val row = w * pixel + 16
            val data = ByteArray(row * h)
            for (y in 0 until h) for (x in 0 until w) {
                val value = (128 + (base(x, y) - 128) * contrast + random.nextInt(-6, 7)).coerceIn(16, 240)
                val offset = y * row + x * pixel
                if (tenBit) {
                    val sample = value shl 2 shl 6
                    data[offset] = sample.toByte()
                    data[offset + 1] = (sample shr 8).toByte()
                } else data[offset] = value.toByte()
            }
            return YuvFrame.Plane(data, row, pixel, tenBit)
        }
        val y = plane(width, height, bytes) { x, row -> 70 + (x * 80 / width) + (row * 20 / height) }
        val u = plane(width / 2, height / 2, bytes) { x, _ -> 118 + x * 20 / width }
        val v = plane(width / 2, height / 2, bytes) { _, row -> 118 + row * 20 / height }
        return YuvFrame(y, u, v, YuvFrame.Crop(0, 0, width, height), 0, false, tenBit, YuvMatrix.BT709)
    }

    private fun generic(frame: Pixels) = object : Pixels {
        override val width = frame.width
        override val height = frame.height
        override fun rgb(x: Int, y: Int) = frame.rgb(x, y)
    }

    @Test
    fun fastModelInputMatchesPerPixelBilinear() {
        for (tenBit in listOf(false, true)) {
            val frame = frame(384, 216, tenBit, 7)
            val transform = Letterbox(384, 140, 141, 52)
            val fast = YoloInput(141, 52).prepare(frame, transform).asFloatBuffer()
            val slow = YoloInput(141, 52).prepare(generic(frame), transform).asFloatBuffer()
            var worst = 0f
            for (i in 0 until fast.limit()) worst = maxOf(worst, abs(fast.get(i) - slow.get(i)))
            // Converting once after blending only differs by rounding: at most two 8-bit levels.
            assertTrue("tenBit=$tenBit worst=${worst * 255}", worst * 255 <= 2.01f)
        }
    }

    @Test
    fun fastModelInputStaysCloseWhenColorsClip() {
        val frame = frame(384, 216, false, 11, contrast = 4)
        val transform = Letterbox(384, 140, 141, 52)
        val fast = YoloInput(141, 52).prepare(frame, transform).asFloatBuffer()
        val slow = YoloInput(141, 52).prepare(generic(frame), transform).asFloatBuffer()
        var total = 0.0
        for (i in 0 until fast.limit()) total += abs(fast.get(i) - slow.get(i)) * 255
        // Pixels that blend clipped and unclipped samples can move more, but the image barely changes.
        assertTrue("mean=${total / fast.limit()}", total / fast.limit() < 1.5)
    }

    @Test
    fun fastModelInputIsFasterOnAFourKFrame() {
        val frame = frame(3840, 2160, false, 3)
        val transform = Letterbox(3840, 1404, 1408, 512)
        val fast = YoloInput(1408, 512)
        val slow = YoloInput(1408, 512)
        val pixels = generic(frame)
        fun time(block: () -> Unit): Double {
            repeat(2) { block() }
            val start = System.nanoTime()
            repeat(3) { block() }
            return (System.nanoTime() - start) / 3e6
        }
        val fastMs = time { fast.prepare(frame, transform) }
        val slowMs = time { slow.prepare(pixels, transform) }
        println("4K model input: per-pixel %.1f ms, row path %.1f ms".format(slowMs, fastMs))
        assertTrue("fast=$fastMs slow=$slowMs", fastMs < slowMs)
    }
}
