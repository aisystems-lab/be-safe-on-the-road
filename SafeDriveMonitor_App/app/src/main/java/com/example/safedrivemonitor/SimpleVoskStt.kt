package com.example.safedrivemonitor

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.*
import org.json.JSONObject
import org.vosk.LibVosk
import org.vosk.LogLevel
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File

class SimpleVoskStt(private val ctx: Context, private val assetModelDir: String) {
    private val sampleRate = 16000
    private var model: Model? = null
    private var recognizer: Recognizer? = null
    private var record: AudioRecord? = null
    private var loopJob: Job? = null

    init {
        LibVosk.setLogLevel(LogLevel.INFO)

        // Copy model from assets → internal storage
        val outRoot = File(ctx.filesDir, assetModelDir)
        if (!outRoot.exists()) {
            copyAssetsRec(assetModelDir, outRoot)
        }

        // Load model
        model = Model(outRoot.absolutePath)
        recognizer = Recognizer(model, sampleRate.toFloat())

        // Setup AudioRecord
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf
        )


    }

    fun start(onResult: (String) -> Unit) {
        if (loopJob != null) return
        record?.startRecording()

        loopJob = CoroutineScope(Dispatchers.IO).launch {
            val buf = ShortArray(2048)
            while (isActive) {
                val n: Int = record?.read(buf, 0, buf.size) ?: 0
                if (n > 0 && recognizer?.acceptWaveForm(buf, n) == true) {
                    val res = recognizer?.getResult() ?: "{}"
                    val text = try {
                        JSONObject(res).optString("text", "")
                    } catch (e: Exception) {
                        ""
                    }
                    if (text.isNotBlank()) {
                        withContext(Dispatchers.Main) { onResult(text) }
                        break
                    }
                }
            }
        }
    }

    fun stop() {
        loopJob?.cancel()
        loopJob = null
        record?.stop()
    }

    fun release() {
        stop()
        recognizer?.close()
        model?.close()
        record?.release()
    }

    private fun copyAssetsRec(src: String, dst: File) {
        val items = ctx.assets.list(src) ?: emptyArray()
        if (items.isEmpty()) {
            ctx.assets.open(src).use { input ->
                dst.outputStream().use { output -> input.copyTo(output) }
            }
        } else {
            if (!dst.exists()) dst.mkdirs()
            for (name in items) {
                copyAssetsRec("$src/$name", File(dst, name))
            }
        }
    }
}
