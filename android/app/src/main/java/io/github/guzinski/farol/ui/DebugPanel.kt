package io.github.guzinski.farol.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import io.github.guzinski.farol.FarolController
import io.github.guzinski.farol.Operation
import io.github.guzinski.farol.detect.Backend
import java.util.Locale
import kotlin.math.roundToInt

@Composable
fun DebugPanel(controller: FarolController, close: () -> Unit, disclaimer: () -> Unit) {
    val settings = controller.settings
    val enabled = controller.operation == Operation.IDLE
    Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .55f)).clickable(onClick = close)) {
        Surface(Modifier.align(Alignment.CenterEnd).fillMaxHeight().widthIn(max = 390.dp).fillMaxWidth()) {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("Ajustes de análise", style = MaterialTheme.typography.titleLarge)
                    TextButton(onClick = close) { Text("Fechar") }
                }
                val stats = controller.frame
                Text(stats?.let {
                    "${decimal(it.fps)} fps  •  ${decimal(it.inferenceMs)} ms  •  ${it.width} × ${it.height}"
                } ?: "Aguardando quadros")
                controller.modelMessage?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                if (!enabled) Text("Os ajustes podem ser alterados ao encerrar a sessão.")
                HorizontalDivider()
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Choice(settings.model?.label ?: "Selecionar modelo", controller.models, { it.label }, enabled) {
                        controller.change(settings.copy(model = it))
                    }
                    TextButton(onClick = controller::refreshModels, enabled = enabled) { Text("Buscar modelos") }
                }
                Choice(settings.backend.label, Backend.entries, { it.label }, enabled) { controller.change(settings.copy(backend = it)) }
                Text("Hexagon QNN · etapa 2", style = MaterialTheme.typography.bodySmall)
                Toggle("Análise 4K", settings.use4k, enabled) { controller.change(settings.copy(use4k = it)) }
                SliderSetting("Exposição", settings.exposure, -3f..2f, 14, " EV", enabled) { controller.change(settings.copy(exposure = it)) }
                Toggle("Medir exposição nos semáforos", settings.meter, enabled) { controller.change(settings.copy(meter = it)) }
                SliderSetting("Faixa de detecção", (settings.band * 100).toFloat(), 35f..100f, 64, "%", enabled) {
                    controller.change(settings.copy(band = it.roundToInt() / 100.0))
                }
                HorizontalDivider()
                Toggle("Alertas por voz", settings.speech, enabled) { controller.change(settings.copy(speech = it)) }
                SliderSetting("Velocidade da voz", settings.rate, .5f..1.5f, 9, "×", enabled) { controller.change(settings.copy(rate = it)) }
                Choice(settings.language, listOf("pt-BR", "en"), { it }, enabled) { controller.change(settings.copy(language = it)) }
                controller.voiceMessage?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                TextButton(onClick = disclaimer) { Text("Aviso de segurança") }
            }
        }
    }
}

@Composable
private fun Toggle(label: String, checked: Boolean, enabled: Boolean, change: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.weight(1f))
        Switch(checked, onCheckedChange = change, enabled = enabled)
    }
}

@Composable
private fun SliderSetting(label: String, value: Float, range: ClosedFloatingPointRange<Float>, steps: Int,
    suffix: String, enabled: Boolean, change: (Float) -> Unit) {
    var draft by remember(value) { mutableFloatStateOf(value) }
    Text("$label  ${decimal(draft.toDouble())}$suffix")
    Slider(draft, { draft = it }, valueRange = range, steps = steps, enabled = enabled, onValueChangeFinished = { change(draft) })
}

@Composable
private fun <T> Choice(label: String, choices: List<T>, text: (T) -> String, enabled: Boolean, selected: (T) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Box {
        Button(onClick = { expanded = true }, enabled = enabled && choices.isNotEmpty()) { Text(label, maxLines = 1) }
        DropdownMenu(expanded, onDismissRequest = { expanded = false }) {
            for (choice in choices) DropdownMenuItem(text = { Text(text(choice)) }, onClick = {
                expanded = false
                selected(choice)
            })
        }
    }
}

private fun decimal(value: Double) = String.format(Locale.forLanguageTag("pt-BR"), "%.1f", value)
