package io.github.guzinski.farol.camera

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.DataSpace
import android.media.Image
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import io.github.guzinski.farol.detect.Box
import io.github.guzinski.farol.perception.Pixels
import java.nio.ByteBuffer
import kotlin.math.roundToInt

enum class YuvMatrix(val redV: Double, val greenU: Double, val greenV: Double, val blueU: Double) {
    BT601(1.402, .344136, .714136, 1.772),
    BT709(1.5748, .187324, .468124, 1.8556),
    BT2020(1.4746, .164553, .571353, 1.8814),
}

// One owner per analysis thread. Each frame is consumed before the next frame reuses these arrays.
internal class YuvPlaneBuffers {
    private val planes = Array(3) { ByteArray(0) }

    fun copy(index: Int, source: ByteBuffer): ByteArray {
        val bytes = source.remaining()
        val target = planes[index].takeIf { it.size >= bytes }
            ?: ByteArray(bytes).also { planes[index] = it }
        source.duplicate().get(target, 0, bytes)
        return target
    }
}

class YuvFrame internal constructor(
    private val y: Plane,
    private val u: Plane,
    private val v: Plane,
    private val crop: Crop,
    private val rotation: Int,
    fullRange: Boolean,
    tenBit: Boolean,
    matrix: YuvMatrix,
) : Pixels {
    internal class Crop(val left: Int, val top: Int, val width: Int, val height: Int) {
        constructor(rect: Rect) : this(rect.left, rect.top, rect.width(), rect.height())
    }

    internal class Plane(private val data: ByteArray, private val row: Int, private val pixel: Int, private val tenBit: Boolean) {
        fun at(x: Int, y: Int): Int {
            val offset = y * row + x * pixel
            val low = data[offset].toInt() and 255
            return if (tenBit) ((data[offset + 1].toInt() and 255) shl 2) or (low shr 6) else low
        }
    }

    override val width = if (rotation % 180 == 0) crop.width else crop.height
    override val height = if (rotation % 180 == 0) crop.height else crop.width
    private val sampleScale = if (tenBit) 4 else 1
    private val fullPeak = 256 * sampleScale - 1
    private val yGain = (255.0 * GAIN / if (fullRange) fullPeak else 219 * sampleScale).roundToInt()
    private val yOffset = if (fullRange) 0 else 16 * sampleScale
    private val uvOffset = 128 * sampleScale
    private val uvGain = 255.0 * GAIN / if (fullRange) fullPeak else 224 * sampleScale
    private val redV = (matrix.redV * uvGain).roundToInt()
    private val greenU = (matrix.greenU * uvGain).roundToInt()
    private val greenV = (matrix.greenV * uvGain).roundToInt()
    private val blueU = (matrix.blueU * uvGain).roundToInt()

    override fun rgb(x: Int, y: Int): Int {
        val sourceX = crop.left + when (rotation) {
            90 -> y
            180 -> crop.width - 1 - x
            270 -> crop.width - 1 - y
            else -> x
        }
        val sourceY = crop.top + when (rotation) {
            90 -> crop.height - 1 - x
            180 -> crop.height - 1 - y
            270 -> x
            else -> y
        }
        return pack(this.y.at(sourceX, sourceY), u.at(sourceX / 2, sourceY / 2), v.at(sourceX / 2, sourceY / 2))
    }

    // The YUV to RGB conversion is affine, so blending Y, U and V and converting once matches
    // blending the four converted RGB samples up to clamping and rounding, at a quarter of the cost.
    override fun bilinearRow(y0: Int, y1: Int, fy: Float, x0: IntArray, x1: IntArray, fx: FloatArray, count: Int, out: IntArray) {
        if (rotation != 0) return super.bilinearRow(y0, y1, fy, x0, x1, fx, count, out)
        val top = crop.top + y0
        val bottom = crop.top + y1
        val chromaTop = top / 2
        val chromaBottom = bottom / 2
        val left = crop.left
        for (i in 0 until count) {
            val a = left + x0[i]
            val b = left + x1[i]
            val ca = a / 2
            val cb = b / 2
            val wx = fx[i]
            out[i] = pack(
                blend(y.at(a, top), y.at(b, top), y.at(a, bottom), y.at(b, bottom), wx, fy),
                blend(u.at(ca, chromaTop), u.at(cb, chromaTop), u.at(ca, chromaBottom), u.at(cb, chromaBottom), wx, fy),
                blend(v.at(ca, chromaTop), v.at(cb, chromaTop), v.at(ca, chromaBottom), v.at(cb, chromaBottom), wx, fy),
            )
        }
    }

    private fun blend(a: Int, b: Int, c: Int, d: Int, fx: Float, fy: Float): Int {
        val upper = a + (b - a) * fx
        val lower = c + (d - c) * fx
        return (upper + (lower - upper) * fy).roundToInt()
    }

    private fun pack(lumaSample: Int, uSample: Int, vSample: Int): Int {
        val luma = (lumaSample - yOffset) * yGain
        val cb = uSample - uvOffset
        val cr = vSample - uvOffset
        val red = ((luma + redV * cr + GAIN / 2) shr SHIFT).coerceIn(0, 255)
        val green = ((luma - greenU * cb - greenV * cr + GAIN / 2) shr SHIFT).coerceIn(0, 255)
        val blue = ((luma + blueU * cb + GAIN / 2) shr SHIFT).coerceIn(0, 255)
        return (red shl 16) or (green shl 8) or blue
    }

    companion object {
        private const val SHIFT = 12
        private const val GAIN = 1 shl SHIFT

        @androidx.annotation.OptIn(ExperimentalGetImage::class)
        internal fun camera(image: ImageProxy, buffers: YuvPlaneBuffers): YuvFrame {
            val planes = image.planes.mapIndexed { index, plane ->
                Plane(buffers.copy(index, plane.buffer), plane.rowStride, plane.pixelStride, false)
            }
            val space = image.image?.dataSpace ?: DataSpace.DATASPACE_UNKNOWN
            val matrix = when (space) {
                DataSpace.DATASPACE_BT709 -> YuvMatrix.BT709
                DataSpace.DATASPACE_BT2020, DataSpace.DATASPACE_BT2020_HLG,
                DataSpace.DATASPACE_BT2020_PQ -> YuvMatrix.BT2020
                else -> YuvMatrix.BT601
            }
            val fullRange = space == DataSpace.DATASPACE_JFIF
            return YuvFrame(
                planes[0], planes[1], planes[2], Crop(image.cropRect), image.imageInfo.rotationDegrees,
                fullRange, false, matrix,
            )
        }

        internal fun decoder(image: Image, rotation: Int, fullRange: Boolean, matrix: YuvMatrix, buffers: YuvPlaneBuffers): YuvFrame {
            val tenBit = image.format == ImageFormat.YCBCR_P010
            require(tenBit || image.format == ImageFormat.YUV_420_888) { "Formato YUV indisponível: ${image.format}." }
            val planes = image.planes.mapIndexed { index, plane ->
                Plane(buffers.copy(index, plane.buffer), plane.rowStride, plane.pixelStride, tenBit)
            }
            return YuvFrame(planes[0], planes[1], planes[2], Crop(image.cropRect), rotation, fullRange, tenBit, matrix)
        }
    }
}

fun Pixels.crop(box: Box): Pixels = object : Pixels {
    override val width = box.width
    override val height = box.height
    override fun rgb(x: Int, y: Int) = this@crop.rgb(box.left + x, box.top + y)
}

fun Pixels.thumbnail(): Bitmap {
    val width = minOf(854, this.width)
    val height = (this.height.toLong() * width / this.width).toInt().coerceAtLeast(1)
    val pixels = IntArray(width * height)
    for (y in 0 until height) for (x in 0 until width) {
        pixels[y * width + x] = rgb(x * this.width / width, y * this.height / height) or (0xff shl 24)
    }
    return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
}
