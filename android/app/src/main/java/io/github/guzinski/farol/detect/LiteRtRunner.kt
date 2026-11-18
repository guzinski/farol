package io.github.guzinski.farol.detect

import android.content.Context
import io.github.guzinski.farol.perception.Pixels
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class ModelFile(val name: String, val file: File? = null) {
    val label: String get() = if (file == null) "$name (APK)" else name

    fun map(context: Context): MappedByteBuffer = if (file != null) {
        FileInputStream(file).channel.use { it.map(FileChannel.MapMode.READ_ONLY, 0, it.size()) }
    } else {
        context.assets.openFd("models/$name").use { asset ->
            asset.createInputStream().channel.use {
                it.map(FileChannel.MapMode.READ_ONLY, asset.startOffset, asset.length)
            }
        }
    }

    companion object {
        fun list(context: Context): List<ModelFile> {
            val directories = listOfNotNull(context.externalMediaDirs.firstOrNull(), context.filesDir)
            val external = directories.flatMap { root ->
                val directory = File(root, "models").apply { mkdirs() }
                directory.listFiles { file -> file.isFile && file.extension.equals("tflite", true) }
                    .orEmpty().map { ModelFile(it.name, it) }
            }
            return external.sortedBy { it.name } + context.assets.list("models").orEmpty()
                .filter { it.endsWith(".tflite") }.map { ModelFile(it) }
        }
    }
}

enum class Backend(val label: String) { GPU("GPU · FP16"), CPU("CPU · debug") }

class LiteRtRunner(context: Context, model: ModelFile, engine: Backend) : ModelRunner {
    override val name = model.label
    override val backend = engine.name
    private val data = model.map(context)
    private val delegate = if (engine == Backend.GPU) {
        GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true))
    } else null
    private val interpreter: Interpreter
    override val width: Int
    override val height: Int
    private val input: YoloInput
    private val output: YoloOutput
    override var preprocessMs = 0.0
        private set
    override var inferenceMs = 0.0
        private set

    init {
        val options = Interpreter.Options().setNumThreads(4)
        delegate?.let { options.addDelegate(it) }
        interpreter = try {
            Interpreter(data, options)
        } catch (error: Throwable) {
            delegate?.close()
            throw error
        }
        try {
            require(interpreter.inputTensorCount == 1 && interpreter.outputTensorCount == 1)
            val tensor = interpreter.getInputTensor(0)
            val shape = tensor.shape()
            require(tensor.dataType() == DataType.FLOAT32 && shape.size == 4 && shape[0] == 1 && shape[3] == 3) {
                "Expected an NHWC float32 RGB input. Export YOLO11s with half=True and nms=False."
            }
            height = shape[1]
            width = shape[2]
            val result = interpreter.getOutputTensor(0)
            require(result.dataType() == DataType.FLOAT32) { "Expected a float32 YOLO output." }
            input = YoloInput(width, height)
            output = YoloOutput(result.shape(), width, height)
        } catch (error: Throwable) {
            close()
            throw error
        }
    }

    override fun detect(frame: Pixels, band: Double): List<Detection> {
        val prepareStart = System.nanoTime()
        val transform = Letterbox(frame.width, (frame.height * band).toInt().coerceIn(1, frame.height), width, height)
        val prepared = input.prepare(frame, transform)
        val start = System.nanoTime()
        preprocessMs = (start - prepareStart) / 1e6
        interpreter.run(prepared, output.buffer.rewind())
        inferenceMs = (System.nanoTime() - start) / 1e6
        return output.decode(frame.width, frame.height, transform)
    }

    override fun close() {
        interpreter.close()
        delegate?.close()
    }
}
