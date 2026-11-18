package io.github.guzinski.farol.replay

import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import io.github.guzinski.farol.camera.YuvFrame
import io.github.guzinski.farol.camera.YuvMatrix
import io.github.guzinski.farol.detect.FrameGate
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.CancellationException

class VideoFrameSource(private val context: Context, private val uri: Uri) {
    // Preserve MP4 tracks, HDR metadata and timestamps byte for byte.
    fun saveVideo(target: File, cancelled: () -> Boolean) {
        val partial = File(target.parentFile, "video.tmp")
        try {
            if (context.contentResolver.getType(uri) == "video/mp4") {
                val stream = checkNotNull(context.contentResolver.openInputStream(uri)) { "Não foi possível abrir o vídeo." }
                stream.use { input ->
                    partial.outputStream().buffered().use { output ->
                        val buffer = ByteArray(128 * 1024)
                        while (!cancelled()) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                        }
                    }
                }
            } else muxVideoTrack(partial, cancelled)
            if (cancelled()) throw CancellationException()
            check(partial.renameTo(target)) { "Não foi possível salvar o vídeo." }
        } finally {
            if (partial.exists()) partial.delete()
        }
    }

    private fun muxVideoTrack(partial: File, cancelled: () -> Boolean) {
        val source = MediaExtractor()
        var muxer: MediaMuxer? = null
        try {
            source.setDataSource(context, uri, null)
            val format = selectVideo(source)
            val writer = MediaMuxer(partial.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer = writer
            writer.setOrientationHint(format.getInteger(MediaFormat.KEY_ROTATION, 0))
            val output = writer.addTrack(format)
            writer.start()
            val origin = source.sampleTime
            check(origin >= 0) { "O vídeo está vazio." }
            var buffer = ByteBuffer.allocateDirect(1024 * 1024)
            val info = MediaCodec.BufferInfo()
            while (source.sampleTime >= 0) {
                if (cancelled()) throw CancellationException()
                val bytes = source.sampleSize
                check(bytes in 1..Int.MAX_VALUE.toLong())
                if (buffer.capacity() < bytes) buffer = ByteBuffer.allocateDirect(bytes.toInt())
                buffer.clear()
                val size = source.readSampleData(buffer, 0)
                check(size == bytes.toInt()) { "Quadro do vídeo incompleto." }
                val flags = if ((source.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC) != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                info.set(0, size, source.sampleTime - origin, flags)
                writer.writeSampleData(output, buffer, info)
                source.advance()
            }
            writer.stop()
            muxer = null
            writer.release()
        } finally {
            muxer?.release()
            source.release()
        }
    }

    fun run(file: File, cancelled: () -> Boolean, frame: (Long, YuvFrame) -> Unit): Double {
        val source = MediaExtractor()
        var decoder: MediaCodec? = null
        try {
            source.setDataSource(file.absolutePath)
            val format = selectVideo(source)
            val rotation = format.getInteger(MediaFormat.KEY_ROTATION, 0)
            format.setInteger(MediaFormat.KEY_ROTATION, 0)
            val mime = checkNotNull(format.getString(MediaFormat.KEY_MIME))
            val transfer = format.getInteger(MediaFormat.KEY_COLOR_TRANSFER, 0)
            val main10 = mime == MediaFormat.MIMETYPE_VIDEO_HEVC &&
                format.getInteger(MediaFormat.KEY_PROFILE, 0) == MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10
            val hdr = transfer == MediaFormat.COLOR_TRANSFER_HLG || transfer == MediaFormat.COLOR_TRANSFER_ST2084
            val tenBit = main10 || hdr
            if (hdr) format.setInteger(MediaFormat.KEY_COLOR_TRANSFER_REQUEST, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, if (tenBit) {
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUVP010
            } else MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            val name = MediaCodecList(MediaCodecList.REGULAR_CODECS).findDecoderForFormat(format)
            val codec = if (name != null) MediaCodec.createByCodecName(name) else MediaCodec.createDecoderByType(mime)
            decoder = codec
            codec.configure(format, null, null, 0)
            codec.start()
            val info = MediaCodec.BufferInfo()
            val gate = FrameGate()
            var inputEnded = false
            var lastPts = 0L
            var progressAt = System.nanoTime()
            while (!cancelled()) {
                if (!inputEnded) {
                    val index = codec.dequeueInputBuffer(0)
                    if (index >= 0) {
                        val buffer = checkNotNull(codec.getInputBuffer(index))
                        val size = source.readSampleData(buffer, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputEnded = true
                        } else {
                            codec.queueInputBuffer(index, 0, size, source.sampleTime, 0)
                            source.advance()
                        }
                    }
                }
                val index = codec.dequeueOutputBuffer(info, 5_000)
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) progressAt = System.nanoTime()
                if (index < 0) {
                    check(System.nanoTime() - progressAt < 10_000_000_000L) { "O decodificador parou de responder." }
                    continue
                }
                try {
                    if (info.size > 0) {
                        lastPts = maxOf(lastPts, info.presentationTimeUs)
                        val ns = info.presentationTimeUs * 1_000
                        if (gate.accept(ns)) {
                            val colors = codec.getOutputFormat(index)
                            val standard = colors.getInteger(MediaFormat.KEY_COLOR_STANDARD,
                                format.getInteger(MediaFormat.KEY_COLOR_STANDARD, 0))
                            val matrix = when (standard) {
                                MediaFormat.COLOR_STANDARD_BT709 -> YuvMatrix.BT709
                                MediaFormat.COLOR_STANDARD_BT2020 -> YuvMatrix.BT2020
                                else -> YuvMatrix.BT601
                            }
                            val range = colors.getInteger(MediaFormat.KEY_COLOR_RANGE, format.getInteger(MediaFormat.KEY_COLOR_RANGE, 0))
                            checkNotNull(codec.getOutputImage(index)) { "O decodificador não disponibilizou quadros YUV." }.use { image ->
                                frame(ns, YuvFrame.decoder(image, rotation, range == MediaFormat.COLOR_RANGE_FULL, matrix))
                            }
                        }
                    }
                } finally {
                    codec.releaseOutputBuffer(index, false)
                }
                progressAt = System.nanoTime()
                if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break
            }
            return maxOf(lastPts, format.getLong(MediaFormat.KEY_DURATION, 0)) / 1e6
        } finally {
            decoder?.release()
            source.release()
        }
    }

    private fun selectVideo(source: MediaExtractor): MediaFormat {
        for (index in 0 until source.trackCount) {
            val format = source.getTrackFormat(index)
            if (format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                source.selectTrack(index)
                return format
            }
        }
        error("O arquivo não contém uma faixa de vídeo.")
    }
}
