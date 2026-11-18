package androidx.camera.video;

import android.annotation.SuppressLint;
import android.media.MediaCodec;
import android.media.MediaFormat;
import android.os.ParcelFileDescriptor;

import androidx.camera.video.internal.encoder.EncoderCallback;
import androidx.camera.video.internal.encoder.EncoderImpl;
import androidx.camera.video.internal.encoder.EncodedData;
import androidx.camera.video.internal.encoder.EncodeException;
import androidx.camera.video.internal.encoder.OutputConfig;
import androidx.camera.video.internal.muxer.Media3MuxerImpl;
import androidx.camera.video.internal.muxer.Muxer;
import androidx.camera.video.internal.muxer.MuxerException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.Executor;
import java.util.function.LongConsumer;

/**
 * CameraX 1.6.2 integration: Recorder subtracts the first encoder PTS before muxing.
 * Report the original uptime PTS only after Recorder actually writes the sample.
 * This bridge lives alongside Recorder to use its package-private encoder factory.
 */
@SuppressLint("RestrictedApi")
public final class RecorderClockBridge {
    private RecorderClockBridge() {}

    public static Recorder.Builder attach(Recorder.Builder builder, LongConsumer onVideoSample) {
        ThreadLocal<Long> originalPts = new ThreadLocal<>();
        builder.setVideoEncoderFactory((executor, config, sessionType) ->
            new EncoderImpl(executor, config, sessionType) {
                @Override
                public void setEncoderCallback(EncoderCallback callback, Executor callbackExecutor) {
                    super.setEncoderCallback(new EncoderCallback() {
                        @Override public void onEncodeStart() { callback.onEncodeStart(); }
                        @Override public void onEncodeStop() { callback.onEncodeStop(); }
                        @Override public void onEncodePaused() { callback.onEncodePaused(); }
                        @Override public void onEncodeError(EncodeException error) {
                            callback.onEncodeError(error);
                        }
                        @Override public void onOutputConfigUpdate(OutputConfig config) {
                            callback.onOutputConfigUpdate(config);
                        }
                        @Override public void onEncodedData(EncodedData data) {
                            originalPts.set(data.getPresentationTimeUs());
                            try {
                                callback.onEncodedData(data);
                            } finally {
                                originalPts.remove();
                            }
                        }
                    }, callbackExecutor);
                }
            });
        return builder.setMuxerFactory(format -> {
            if (format != Muxer.MUXER_FORMAT_MPEG_4) {
                throw new IllegalArgumentException("Farol requires MP4 video.");
            }
            return new ClockMuxer(new Media3MuxerImpl(), originalPts, onVideoSample);
        });
    }

    private static final class ClockMuxer implements Muxer {
        private final Muxer muxer;
        private final ThreadLocal<Long> originalPts;
        private final LongConsumer onVideoSample;

        ClockMuxer(Muxer muxer, ThreadLocal<Long> originalPts, LongConsumer onVideoSample) {
            this.muxer = muxer;
            this.originalPts = originalPts;
            this.onVideoSample = onVideoSample;
        }

        @Override public void setOutput(String path, int format) throws IOException { muxer.setOutput(path, format); }
        @Override public void setOutput(ParcelFileDescriptor fd, int format) throws IOException {
            muxer.setOutput(fd, format);
        }
        @Override public void setOrientationDegrees(int degrees) { muxer.setOrientationDegrees(degrees); }
        @Override public void setLocation(double lat, double lon) { muxer.setLocation(lat, lon); }
        @Override public void setCaptureFps(int fps) { muxer.setCaptureFps(fps); }
        @Override public int addTrack(MediaFormat format) throws MuxerException { return muxer.addTrack(format); }
        @Override public void start() throws MuxerException { muxer.start(); }
        @Override public void stop() throws MuxerException { muxer.stop(); }
        @Override public void release() { muxer.release(); }
        @Override public boolean isInterruptionResilient() { return muxer.isInterruptionResilient(); }

        @Override public void writeSampleData(int track, ByteBuffer data, MediaCodec.BufferInfo info)
                throws MuxerException {
            muxer.writeSampleData(track, data, info);
            Long timestamp = originalPts.get();
            if (timestamp != null && info.size > 0) onVideoSample.accept(timestamp);
        }
    }
}
