package io.github.guzinski.farol

import io.github.guzinski.farol.perception.LightColor
import io.github.guzinski.farol.perception.Pixels
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import javax.imageio.ImageIO

class ParityTest {
    @Test
    fun matchesDesktopTrafficLightCrops() {
        val fixtures = JSONArray(resource("/parity/fixtures.json").bufferedReader().use { it.readText() })
        assertTrue(fixtures.length() >= 13)
        for (index in 0 until fixtures.length()) {
            val expected = fixtures.getJSONObject(index)
            val name = expected.getString("file")
            val image = resource("/parity/$name").use { ImageIO.read(it) }
            val actual = LightColor.read(object : Pixels {
                override val width = image.width
                override val height = image.height
                override fun rgb(x: Int, y: Int) = image.getRGB(x, y)
            })
            val counts = expected.optJSONArray("lit_pixel_counts")
            if (counts == null) assertEquals(name, null, actual.counts)
            else {
                val colors = counts.getJSONObject(0)
                val read = checkNotNull(actual.counts)
                assertEquals(name, colors.getInt("red"), read.red)
                assertEquals(name, colors.getInt("yellow"), read.yellow)
                assertEquals(name, colors.getInt("green"), read.green)
                assertEquals(name, counts.getInt(1), read.area)
            }
            val thirds = expected.optJSONArray("thirds_brightness")
            if (thirds == null) assertEquals(name, null, actual.thirds)
            else for (third in 0..2) {
                assertEquals(name, thirds.getDouble(third), checkNotNull(actual.thirds)[third], 0.5)
            }
            assertEquals(name, expected.get("lamp_by_position") as? String, actual.lampByPosition?.key)
            assertEquals(name, expected.getBoolean("seen_from_behind"), actual.seenFromBehind)
            assertEquals(name, expected.get("traffic_light_color") as? String, actual.color?.key)
        }
    }

    private fun resource(path: String) = checkNotNull(javaClass.getResourceAsStream(path)) { "Missing $path" }
}
