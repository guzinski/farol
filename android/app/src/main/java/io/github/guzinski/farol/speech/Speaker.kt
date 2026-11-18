package io.github.guzinski.farol.speech

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

class Speaker(context: Context, private val onMessage: (String?) -> Unit) : AutoCloseable {
    private val main = Handler(Looper.getMainLooper())
    private val audio = context.getSystemService(AudioManager::class.java)
    private val attributes = AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()
    private val focus = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
        .setAudioAttributes(attributes)
        .setOnAudioFocusChangeListener { if (it == AudioManager.AUDIOFOCUS_LOSS) stop() }
        .build()
    private var initialized = false
    private var ready = false
    private var closed = false
    private var sequence = 0L
    private var current: String? = null
    private var language = "pt-BR"
    private val tts = TextToSpeech(context.applicationContext) { result ->
        main.post {
            if (closed) return@post
            initialized = result == TextToSpeech.SUCCESS
            if (initialized) configureLanguage()
            else onMessage("Ative um mecanismo nas configurações de texto para fala do Android.")
        }
    }

    init {
        tts.setAudioAttributes(attributes)
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String) = Unit
            override fun onDone(utteranceId: String) = finished(utteranceId)
            @Deprecated("Android callback")
            override fun onError(utteranceId: String) = finished(utteranceId)
            override fun onStop(utteranceId: String, interrupted: Boolean) = finished(utteranceId)
        })
    }

    fun setLanguage(tag: String) {
        language = tag
        if (initialized) configureLanguage()
    }

    private fun configureLanguage() {
        val locale = if (language == "pt-BR") Locale.forLanguageTag("pt-BR") else Locale.US
        val result = tts.setLanguage(locale)
        ready = result >= TextToSpeech.LANG_COUNTRY_AVAILABLE &&
            tts.voice?.features?.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED) != true
        onMessage(if (ready) null else "Instale uma voz $language nas configurações de texto para fala do Android.")
    }

    fun speak(text: String, rate: Float) {
        if (!ready || closed) return
        if (current == null && audio.requestAudioFocus(focus) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) return
        tts.setSpeechRate(rate)
        val id = (++sequence).toString()
        current = id
        if (tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, id) == TextToSpeech.ERROR) finished(id)
    }

    private fun finished(id: String) {
        main.post {
            if (current == id) {
                current = null
                audio.abandonAudioFocusRequest(focus)
            }
        }
    }

    fun stop() {
        tts.stop()
        current = null
        audio.abandonAudioFocusRequest(focus)
    }

    override fun close() {
        closed = true
        stop()
        tts.shutdown()
    }
}
