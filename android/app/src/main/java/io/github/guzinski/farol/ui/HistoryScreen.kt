package io.github.guzinski.farol.ui

import android.media.MediaPlayer
import android.widget.MediaController
import android.widget.VideoView
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.session.SavedSession
import io.github.guzinski.farol.session.SessionDetails
import io.github.guzinski.farol.session.SessionHistory
import io.github.guzinski.farol.session.SessionRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

@Composable
fun HistoryScreen(close: () -> Unit) {
    val context = LocalContext.current.applicationContext
    var sessions by remember { mutableStateOf<List<SavedSession>?>(null) }
    var selected by remember { mutableStateOf<SavedSession?>(null) }
    val back = { if (selected == null) close() else selected = null }

    BackHandler(onBack = back)
    LaunchedEffect(context) {
        sessions = withContext(Dispatchers.IO) { SessionHistory.list(SessionRecorder.root(context)) }
    }

    Column(Modifier.fillMaxSize().padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Text(selected?.title() ?: "Histórico", style = MaterialTheme.typography.headlineSmall)
            TextButton(onClick = back) { Text(if (selected == null) "Fechar" else "Voltar ao histórico") }
        }
        val session = selected
        if (session == null) {
            SessionList(sessions) { selected = it }
        } else {
            SessionReview(session)
        }
    }
}

@Composable
private fun SessionList(sessions: List<SavedSession>?, select: (SavedSession) -> Unit) {
    when {
        sessions == null -> Text("Carregando sessões…")
        sessions.isEmpty() -> Text("Nenhuma sessão salva. Inicie uma sessão na câmera ou reproduza um vídeo.")
        else -> LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(sessions, key = { it.id }) { session ->
                Card(onClick = { select(session) }, modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(session.title(), style = MaterialTheme.typography.titleMedium)
                        val facts = listOfNotNull(
                            session.duration?.let(::formatTime),
                            session.frameCount?.let { "$it quadros analisados" },
                            session.error?.let { "Com erro" },
                            "Sessão incompleta".takeUnless { session.finalized },
                        )
                        if (facts.isNotEmpty()) Text(facts.joinToString("  ·  "), style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }
}

@Composable
private fun SessionReview(session: SavedSession) {
    var details by remember(session.id) { mutableStateOf<SessionDetails?>(null) }
    var videoView by remember(session.id) { mutableStateOf<VideoView?>(null) }
    var player by remember(session.id) { mutableStateOf<MediaPlayer?>(null) }
    var videoError by remember(session.id) { mutableStateOf(false) }
    val video = session.video
    val hasVideo = remember(video) { video.isFile && video.length() > 0 }

    LaunchedEffect(session) {
        details = withContext(Dispatchers.IO) { SessionHistory.details(session) }
    }

    Row(Modifier.fillMaxSize(), horizontalArrangement = Arrangement.spacedBy(18.dp)) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Vídeo", style = MaterialTheme.typography.titleMedium)
            if (hasVideo && !videoError) {
                Box(Modifier.fillMaxWidth().aspectRatio(16f / 9f).background(Color.Black)) {
                    AndroidView(
                        factory = { context ->
                            VideoView(context).apply {
                                setMediaController(MediaController(context))
                                setOnPreparedListener {
                                    player = it
                                    seekTo(1)
                                }
                                setOnErrorListener { _, _, _ ->
                                    player = null
                                    videoError = true
                                    true
                                }
                                setVideoPath(video.absolutePath)
                                videoView = this
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                        onRelease = {
                            it.stopPlayback()
                            videoView = null
                            player = null
                        },
                    )
                }
                Button(onClick = {
                    videoView?.let { view -> if (view.isPlaying) view.pause() else view.start() }
                }, enabled = player != null) { Text("Reproduzir / pausar") }
            } else {
                Text(if (videoError) "Este vídeo não pôde ser reproduzido." else "A sessão não tem um vídeo salvo.")
            }
        }

        val record = details
        LazyColumn(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            item {
                Text("Análise", style = MaterialTheme.typography.titleMedium)
                val frameCount = record?.detections?.frames ?: session.frameCount
                val info = listOfNotNull(
                    frameCount?.let { "$it quadros analisados" },
                    session.fps?.let { String.format(Locale.forLanguageTag("pt-BR"), "%.1f fps", it) },
                    session.resolution,
                )
                if (info.isNotEmpty()) Text(info.joinToString("  ·  "))
                val model = listOfNotNull(session.model, session.backend).joinToString(" · ")
                if (model.isNotEmpty()) Text(model, style = MaterialTheme.typography.bodySmall)
                session.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                if (!session.finalized) Text("Sessão incompleta. Parte dos registros pode faltar.")
            }
            item {
                val detections = record?.detections
                when {
                    record == null -> Text("Carregando registros…")
                    detections == null -> Text("Registro de detecções indisponível.")
                    detections.frames == 0 -> Text("Nenhum quadro analisado foi salvo.")
                    else -> {
                        Text("Semáforos ${detections.lightTracks}  ·  Placas de pare ${detections.stopTracks}")
                        if (detections.lightObservations > 0) {
                            val count = detections.colors.values.sum()
                            Text("$count de ${detections.lightObservations} leituras com cor")
                            val colors = LampColor.entries.mapNotNull { color ->
                                detections.colors[color]?.let { "${label(color)} $it" }
                            }
                            if (colors.isNotEmpty()) Text(colors.joinToString("  ·  "), style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
                if (detections != null && detections.unreadableLines > 0) {
                    Text("${detections.unreadableLines} linhas de detecção não puderam ser lidas.")
                }
            }
            item {
                val alerts = record?.alerts
                Text(alerts?.let { "Alertas (${it.size})" } ?: "Alertas", style = MaterialTheme.typography.titleMedium)
                if (record != null && alerts == null) Text("Registro de alertas indisponível.")
                if (alerts?.isEmpty() == true) Text("Nenhum alerta emitido.")
                if (!alerts.isNullOrEmpty() && player != null) Text("Toque em um alerta para ir ao trecho no vídeo.")
            }
            items(record?.alerts.orEmpty()) { alert ->
                val time = formatTime(alert.seconds)
                if (player != null) {
                    TextButton(onClick = {
                        player?.seekTo((alert.seconds * 1_000).toLong(), MediaPlayer.SEEK_CLOSEST)
                    }) { Text("$time  ${alert.text}") }
                } else Text("$time  ${alert.text}")
            }
        }
    }
}

private fun SavedSession.title(): String {
    val name = when (mode) {
        "drive" -> "Trajeto"
        "replay" -> "Reprodução"
        else -> "Sessão"
    }
    val date = id.substringBefore('_').split('-').reversed().joinToString("/")
    val time = id.substringAfter('_').replace('-', ':')
    return "$name · $date $time"
}

private fun label(color: LampColor): String = when (color) {
    LampColor.RED -> "Vermelho"
    LampColor.YELLOW -> "Amarelo"
    LampColor.GREEN -> "Verde"
}
