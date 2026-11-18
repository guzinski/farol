package io.github.guzinski.farol.ui

import android.graphics.Paint
import android.graphics.Typeface
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import io.github.guzinski.farol.FarolController
import io.github.guzinski.farol.Operation
import io.github.guzinski.farol.detect.Analysis
import io.github.guzinski.farol.detect.STOP_SIGN
import io.github.guzinski.farol.i18n.Translator
import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.perception.Position

private val Red = Color(0xFFFF5E62)
private val Yellow = Color(0xFFFFD34D)
private val Green = Color(0xFF58EF94)
private val Panel = Color(0xE5182224)
private val scheme = darkColorScheme(
    primary = Color(0xFF9DE9C0), onPrimary = Color(0xFF082B19),
    surface = Color(0xFF172123), background = Color(0xFF090D0E),
)

@Composable
fun FarolApp(
    controller: FarolController,
    accepted: Boolean,
    cameraGranted: Boolean,
    accept: () -> Unit,
    requestCamera: () -> Unit,
    chooseVideo: () -> Unit,
) {
    var debug by remember { mutableStateOf(false) }
    var showDisclaimer by remember { mutableStateOf(false) }
    MaterialTheme(colorScheme = scheme) {
        val op = controller.operation
        BackHandler(enabled = debug || controller.replay) {
            if (debug) debug = false
            else if (op == Operation.REPLAYING) controller.stopReplay()
            else if (op == Operation.IDLE) controller.returnToDrive()
        }
        Box(Modifier.fillMaxSize().background(Color(0xFF090D0E))) {
            if (accepted) {
                DriveScreen(controller, cameraGranted, requestCamera, chooseVideo) { debug = true }
                if (debug) DebugPanel(controller, close = { debug = false }, disclaimer = { showDisclaimer = true })
            }
        }
        if (!accepted || showDisclaimer) {
            AlertDialog(
                onDismissRequest = { if (accepted) showDisclaimer = false },
                title = { Text("Farol é um experimento") },
                text = { Text("O aplicativo pode errar ou deixar de emitir alertas. Ele não é um dispositivo de segurança. Não dependa dele para dirigir. Quem dirige não deve operar o telefone com o veículo em movimento.") },
                confirmButton = {
                    Button(onClick = { if (accepted) showDisclaimer = false else accept() }) {
                        Text(if (accepted) "Fechar" else "Entendi")
                    }
                },
            )
        }
    }
}

@Composable
private fun DriveScreen(
    controller: FarolController,
    cameraGranted: Boolean,
    requestCamera: () -> Unit,
    chooseVideo: () -> Unit,
    openDebug: () -> Unit,
) {
    val frame = controller.frame
    val replay = controller.replay
    val op = controller.operation
    val app = LocalContext.current.applicationContext
    val language = controller.settings.language
    val translator = remember(language) { Translator.load(app.assets, language) }

    BoxWithConstraints(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        val ratio = if (replay && frame != null) frame.width.toFloat() / frame.height else 16f / 9f
        val videoSize = if (maxWidth / maxHeight > ratio) Modifier.fillMaxHeight().aspectRatio(ratio, true)
            else Modifier.fillMaxWidth().aspectRatio(ratio)
        Box(videoSize.clipToBounds().background(Color.Black)) {
            if (replay) {
                controller.videoFrame?.let {
                    Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
                }
            } else if (cameraGranted) {
                AndroidView(factory = { controller.preview }, modifier = Modifier.fillMaxSize())
            }
            frame?.let { Overlay(it, translator) }
        }

        val message = when {
            controller.sessionMessage != null -> controller.sessionMessage
            !replay && controller.cameraMessage != null -> controller.cameraMessage
            controller.modelMessage != null -> controller.modelMessage
            controller.modelBusy -> "Carregando o modelo…"
            replay && frame == null && op == Operation.REPLAYING -> "Preparando o vídeo…"
            else -> null
        }
        Column(Modifier.align(Alignment.TopStart).padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(Modifier.background(Panel, RoundedCornerShape(14.dp)).padding(12.dp), horizontalArrangement = Arrangement.spacedBy(18.dp)) {
                Position.entries.forEach { position ->
                    val state = frame?.lights?.getOrNull(position.ordinal)
                    val label = translator("position.${position.key}")
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Box(Modifier.size(30.dp).background(lampColor(state, Color(0xFF455356)), CircleShape)
                            .semantics { contentDescription = "$label, ${state?.let { translator("color.${it.key}") } ?: "—"}" })
                        Text(label, color = Color.White, style = MaterialTheme.typography.labelMedium)
                    }
                }
            }
            if (op == Operation.RECORDING) Tag("● Gravando", Red)
            if (replay) Tag(frame?.let { "Reprodução · ${formatTime(it.time)}" } ?: "Reprodução")
        }

        Row(Modifier.align(Alignment.TopEnd).padding(12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (replay) Button(onClick = controller::returnToDrive, enabled = op == Operation.IDLE) { Text("Câmera") }
            else Button(onClick = chooseVideo, enabled = op == Operation.IDLE && controller.modelReady) { Text("Reproduzir vídeo") }
            Button(onClick = openDebug) { Text("Ajustes") }
        }

        if (!replay && !cameraGranted) {
            Button(onClick = requestCamera) { Text("Permitir câmera") }
        } else if (message != null && controller.lastAlert == null) {
            Box(Modifier.align(Alignment.Center).widthIn(max = 500.dp)) { Tag(message) }
        }
        Column(Modifier.align(Alignment.BottomCenter).padding(18.dp), horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            controller.lastAlert?.let {
                Text(it, modifier = Modifier.widthIn(max = 640.dp).background(Color.Black, RoundedCornerShape(10.dp)).padding(12.dp),
                    color = Color.White, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            }
            when (op) {
                Operation.IDLE -> if (!replay) {
                    Button(onClick = controller::startDrive, enabled = controller.cameraReady && controller.modelReady) {
                        Text("Iniciar sessão", Modifier.padding(horizontal = 12.dp, vertical = 5.dp))
                    }
                }
                Operation.STARTING, Operation.RECORDING ->
                    Button(onClick = controller::stopDrive, colors = ButtonDefaults.buttonColors(containerColor = Red, contentColor = Color.Black)) {
                        Text(if (op == Operation.STARTING) "Cancelar início" else "Encerrar sessão", Modifier.padding(5.dp))
                    }
                Operation.REPLAYING -> Button(onClick = controller::stopReplay) { Text("Parar reprodução") }
                Operation.SAVING -> Tag("Salvando sessão…")
            }
        }
    }
}

@Composable
private fun Tag(text: String, color: Color = Color.White) {
    Text(text, Modifier.background(Panel, RoundedCornerShape(8.dp)).padding(horizontal = 12.dp, vertical = 8.dp), color = color)
}

private fun formatTime(time: Double): String {
    val seconds = time.toInt().coerceAtLeast(0)
    return "${seconds / 60}:${(seconds % 60).toString().padStart(2, '0')}"
}

private fun lampColor(lamp: LampColor?, unknown: Color = Color.White): Color = when (lamp) {
    LampColor.RED -> Red
    LampColor.YELLOW -> Yellow
    LampColor.GREEN -> Green
    null -> unknown
}

@Composable
private fun Overlay(frame: Analysis, translation: Translator) {
    val paint = remember { Paint(Paint.ANTI_ALIAS_FLAG).also { it.typeface = Typeface.DEFAULT_BOLD } }
    Canvas(Modifier.fillMaxSize()) {
        val xScale = size.width / frame.width
        val yScale = size.height / frame.height
        paint.textSize = 13.dp.toPx()
        val gap = 4.dp.toPx()
        for (result in frame.detections) {
            val box = result.box
            val color = if (result.cls == STOP_SIGN) Red else lampColor(result.color)
            val x = box.left * xScale
            val y = box.top * yScale
            drawRect(color, Offset(x, y), Size(box.width * xScale, box.height * yScale), style = Stroke(2.dp.toPx()))
            val label = if (result.cls == STOP_SIGN) translation("label.stop") else {
                listOfNotNull(result.color?.let { translation("color.${it.key}") },
                    result.position?.let { translation("position.${it.key}") }).joinToString(" · ").ifBlank { "?" }
            }
            val width = paint.measureText(label) + gap * 2
            val left = x.coerceAtMost((size.width - width).coerceAtLeast(0f))
            val textBottom = y.coerceAtLeast(paint.textSize + gap * 2)
            drawRect(Color.Black.copy(alpha = .85f), Offset(left, textBottom - paint.textSize - gap * 2),
                Size(width, paint.textSize + gap * 2))
            paint.color = color.toArgb()
            drawContext.canvas.nativeCanvas.drawText(label, left + gap, textBottom - gap, paint)
        }
    }
}
