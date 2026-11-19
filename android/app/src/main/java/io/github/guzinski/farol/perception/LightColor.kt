package io.github.guzinski.farol.perception

import kotlin.math.roundToInt

enum class LampColor(val key: String) { RED("red"), YELLOW("yellow"), GREEN("green") }

interface Pixels {
    val width: Int
    val height: Int
    fun rgb(x: Int, y: Int): Int

    /**
     * Writes [count] bilinear samples between rows [y0] and [y1] into [out] as packed RGB.
     * Sample i blends columns x0[i] and x1[i] with weight fx[i], and the rows with weight [fy].
     */
    fun bilinearRow(y0: Int, y1: Int, fy: Float, x0: IntArray, x1: IntArray, fx: FloatArray, count: Int, out: IntArray) {
        for (i in 0 until count) {
            val topLeft = rgb(x0[i], y0)
            val topRight = rgb(x1[i], y0)
            val bottomLeft = rgb(x0[i], y1)
            val bottomRight = rgb(x1[i], y1)
            var packed = 0
            for (shift in 16 downTo 0 step 8) {
                val a = (topLeft shr shift) and 255
                val b = (topRight shr shift) and 255
                val c = (bottomLeft shr shift) and 255
                val d = (bottomRight shr shift) and 255
                val top = a + (b - a) * fx[i]
                val bottom = c + (d - c) * fx[i]
                packed = packed or ((top + (bottom - top) * fy).roundToInt() shl shift)
            }
            out[i] = packed
        }
    }
}

data class LitCounts(val red: Int, val yellow: Int, val green: Int, val area: Int)
data class LightRead(
    val counts: LitCounts?,
    val thirds: List<Double>?,
    val lampByPosition: LampColor?,
    val seenFromBehind: Boolean,
    val color: LampColor?,
)

object LightColor {
    private const val MIN_LIT_HEIGHT = 8
    private const val MIN_BRIGHTNESS_HEIGHT = 6
    private const val VERTICAL_ASPECT = 2
    private const val LIT_SATURATION = 90
    private const val LIT_VALUE = 160
    private const val RED_LOW = 10
    private const val RED_HIGH = 168
    private val YELLOW = 14..34
    private val GREEN = 42..98
    private val HOUSING_HUE = 12..35
    private const val HOUSING_SATURATION = 40
    private const val HOUSING_VALUE = 90
    private const val HOUSING_FRACTION = .25
    private const val DARK_FRACTION = .25
    private const val MIN_LIT_PIXELS = 4
    private const val MIN_LIT_FRACTION = .02
    private const val BRIGHTNESS_GAP = 25
    private const val STOP_SATURATION = 80
    private const val STOP_VALUE = 60
    private const val FAINT_SATURATION = 35
    private const val FAINT_VALUE = 100
    private const val FAINT_FRACTION = .08
    private const val FAINT_SPREAD = 4

    // OpenCV's 8-bit conversion uses fixed-point reciprocal tables.
    private const val HSV_SHIFT = 12
    private const val ROUND = 1 shl (HSV_SHIFT - 1)
    private val saturationScale = IntArray(256) { if (it == 0) 0 else ((255 shl HSV_SHIFT).toDouble() / it).roundToInt() }
    private val hueScale = IntArray(256) { if (it == 0) 0 else ((180 shl HSV_SHIFT).toDouble() / (6 * it)).roundToInt() }

    private inline fun withHsv(rgb: Int, consume: (Int, Int, Int) -> Unit) {
        val red = (rgb shr 16) and 255
        val green = (rgb shr 8) and 255
        val blue = rgb and 255
        val value = maxOf(red, green, blue)
        val delta = value - minOf(red, green, blue)
        val saturation = (delta * saturationScale[value] + ROUND) shr HSV_SHIFT
        val sector = when (value) {
            red -> green - blue
            green -> blue - red + 2 * delta
            else -> red - green + 4 * delta
        }
        val hue = (sector * hueScale[delta] + ROUND) shr HSV_SHIFT
        consume(if (hue < 0) hue + 180 else hue, saturation, value)
    }

    private fun red(hue: Int) = hue < RED_LOW || hue > RED_HIGH

    fun read(crop: Pixels): LightRead {
        val width = crop.width
        val height = crop.height
        val area = width * height
        if (area == 0) return LightRead(null, null, null, false, null)
        val vertical = height >= VERTICAL_ASPECT * width
        val bounds = intArrayOf(height / 3, height * 2 / 3)
        val rows = IntArray(3) { height / 3 + if (it < height % 3) 1 else 0 }
        val lit = IntArray(3)
        val brightness = LongArray(3)
        var housing = 0
        var dark = 0
        var faintBottom = 0
        var faintUpper = 0
        for (y in 0 until height) {
            val lamp = if (y < bounds[0]) 0 else if (y < bounds[1]) 1 else 2
            val third = if (y < rows[0]) 0 else if (y < rows[0] + rows[1]) 1 else 2
            for (x in 0 until width) withHsv(crop.rgb(x, y)) { h, s, v ->
                brightness[third] += v.toLong()
                if (v < HOUSING_VALUE) dark++
                if (h in HOUSING_HUE && s > HOUSING_SATURATION && v > HOUSING_VALUE) housing++
                if (vertical && h in GREEN && s >= FAINT_SATURATION && v >= FAINT_VALUE) {
                    if (lamp == 2) faintBottom++ else faintUpper++
                }
                if (s > LIT_SATURATION && v > LIT_VALUE) {
                    if (red(h) && (!vertical || lamp == 0)) lit[0]++
                    if (h in YELLOW && (!vertical || lamp == 1)) lit[1]++
                    if (h in GREEN && (!vertical || lamp == 2)) lit[2]++
                }
            }
        }
        val behind = housing.toDouble() / area > HOUSING_FRACTION && dark.toDouble() / area < DARK_FRACTION
        val thirds = if (height >= MIN_BRIGHTNESS_HEIGHT) {
            brightness.indices.map { (brightness[it].toFloat() / (rows[it] * width)).toDouble() }
        } else null
        val lamp = if (vertical && thirds != null) {
            val ranked = thirds.indices.sortedByDescending { thirds[it] }
            LampColor.entries[ranked[0]].takeIf { thirds[ranked[0]] - thirds[ranked[1]] >= BRIGHTNESS_GAP }
        } else null
        val counts = if (height >= MIN_LIT_HEIGHT) LitCounts(lit[0], lit[1], lit[2], area) else null
        val minimum = maxOf(MIN_LIT_PIXELS.toDouble(), MIN_LIT_FRACTION * area)
        val litColor = when {
            counts == null -> null
            // In direct sun the yellow frame around the face of a front-facing light also passes
            // the housing test. Its yellow could still be the housing, but enough lit red or green
            // that beats the yellow or matches the brighter lamp can only come from a lit lamp.
            behind -> {
                val candidate = if (lit[2] > lit[0]) LampColor.GREEN else LampColor.RED
                val count = lit[candidate.ordinal]
                candidate.takeIf { count >= minimum && (count > lit[1] || lamp == candidate) }
            }
            else -> {
                val best = lit.indices.maxBy { lit[it] }
                if (lit[best] < minimum) lamp else LampColor.entries[best]
            }
        }
        // A green lamp the camera renders almost white, such as a green arrow in sun, keeps only
        // a faint green rim. Leaves or sky behind the box spread over all of it instead.
        val faintArea = (height - bounds[1]) * width
        val color = litColor ?: LampColor.GREEN.takeIf {
            counts != null && vertical && faintBottom >= maxOf(MIN_LIT_PIXELS.toDouble(), FAINT_FRACTION * faintArea) &&
                FAINT_SPREAD * faintUpper <= faintBottom
        }
        return LightRead(counts, thirds, lamp, behind, color)
    }

    fun redFraction(crop: Pixels): Double {
        val area = crop.width * crop.height
        if (area == 0) return 0.0
        var count = 0
        for (y in 0 until crop.height) for (x in 0 until crop.width) {
            withHsv(crop.rgb(x, y)) { h, s, v ->
                if (red(h) && s > STOP_SATURATION && v > STOP_VALUE) count++
            }
        }
        return count.toDouble() / area
    }
}
