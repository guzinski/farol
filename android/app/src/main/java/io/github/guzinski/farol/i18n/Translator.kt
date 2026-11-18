package io.github.guzinski.farol.i18n

import android.content.res.AssetManager
import io.github.guzinski.farol.rules.Alert
import org.json.JSONObject

class Translator(english: String, selected: String = english) {
    private val fallback = JSONObject(english)
    private val messages = JSONObject(selected)
    private val parameter = Regex("\\{([^{}]+)\\}")

    operator fun invoke(key: String, params: Map<String, String> = emptyMap()): String {
        val template = if (messages.has(key)) messages.getString(key) else fallback.getString(key)
        return parameter.replace(template) { params.getValue(it.groupValues[1]) }
    }

    fun alert(alert: Alert): String {
        val lights = alert.lights ?: return invoke(alert.key)
        return invoke(alert.key, mapOf("lights" to lights.joinToString(", ") {
            invoke("light.at", mapOf(
                "position" to invoke("position.${it.position.key}"),
                "color" to invoke("color.${it.color.key}"),
            ))
        }))
    }

    companion object {
        fun load(assets: AssetManager, language: String): Translator {
            fun read(tag: String) = assets.open("locales/$tag.json").bufferedReader().use { it.readText() }
            val english = read("en")
            return Translator(english, if (language == "en") english else read(language))
        }
    }
}
