package com.example.safedrivemonitor

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.*
import okhttp3.*
import com.google.gson.Gson
import java.io.IOException
import java.util.concurrent.TimeUnit
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * AssistantFragment
 * -----------------
 * Supports 9 languages via Android SpeechRecognizer (BCP-47 locale).
 *
 * Language-aware query strategy (no backend changes):
 *   Each supported language has a native-language instruction prefix.
 *   When the selected language is not English, the prefix is prepended
 *   so the LLM responds in the correct language.
 */
class AssistantFragment : Fragment() {

    // ── Language prefix map (BCP-47 base → native instruction) ────────────
    private val langPrefixes = mapOf(
        "es" to "Por favor responde en español. Pregunta: ",
        "hi" to "कृपया हिंदी में उत्तर दें। प्रश्न: ",
        "fr" to "Veuillez répondre en français. Question : ",
        "mr" to "कृपया मराठीत उत्तर द्या. प्रश्न: ",
        "ja" to "日本語で答えてください。質問：",
        "de" to "Bitte antworte auf Deutsch. Frage: ",
        "zh" to "请用中文回答。问题：",
        "ru" to "Пожалуйста, ответьте по-русски. Вопрос: "
        // "en" → no prefix needed
    )

    // ── UI refs ────────────────────────────────────────────────────────────
    private lateinit var rvChat: RecyclerView
    private lateinit var btnMic: View
    private lateinit var tvListeningState: TextView
    private lateinit var tvLanguageBadge: TextView

    private lateinit var adapter: ChatAdapter
    private val messages = mutableListOf<ChatMessage>()

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    private val gson  = Gson()
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false

    // ── Lifecycle ──────────────────────────────────────────────────────────
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_assistant, container, false)
        rvChat           = view.findViewById(R.id.rvChat)
        btnMic           = view.findViewById(R.id.btnMic)
        tvListeningState = view.findViewById(R.id.tvListeningState)
        tvLanguageBadge  = view.findViewById(R.id.tvLanguageBadge)

        adapter = ChatAdapter(messages)
        rvChat.adapter = adapter
        rvChat.layoutManager = LinearLayoutManager(requireContext())

        initSpeechRecognizer()
        updateLanguageBadge()

        btnMic.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN         -> { startListening(); true }
                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL       -> { stopListening();  true }
                else                            -> false
            }
        }
        return view
    }

    override fun onResume() {
        super.onResume()
        updateLanguageBadge()
    }

    // ── Speech Recognizer ──────────────────────────────────────────────────
    private fun initSpeechRecognizer() {
        speechRecognizer?.destroy()
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(requireContext())
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(p: Bundle?) {
                scope.launch { setState(ListenState.LISTENING) }
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rms: Float) {}
            override fun onBufferReceived(buf: ByteArray?) {}
            override fun onEndOfSpeech() {
                scope.launch { setState(ListenState.PROCESSING) }
            }
            override fun onError(error: Int) {
                scope.launch {
                    isListening = false
                    btnMic.isSelected = false
                    setState(ListenState.IDLE)
                    if (error != SpeechRecognizer.ERROR_CLIENT &&
                        error != SpeechRecognizer.ERROR_RECOGNIZER_BUSY) {
                        val msg = when (error) {
                            SpeechRecognizer.ERROR_NO_MATCH        -> getString(R.string.stt_no_match)
                            SpeechRecognizer.ERROR_NETWORK,
                            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> getString(R.string.stt_network_error)
                            else -> getString(R.string.stt_generic_error)
                        }
                        Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
                    }
                }
            }
            override fun onResults(results: Bundle?) {
                val text = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull().orEmpty().trim()
                scope.launch {
                    isListening = false
                    btnMic.isSelected = false
                    setState(ListenState.IDLE)
                    if (text.isNotBlank()) handleQuery(text)
                }
            }
            override fun onPartialResults(p: Bundle?) {}
            override fun onEvent(t: Int, p: Bundle?) {}
        })
    }

    private fun startListening() {
        if (ActivityCompat.checkSelfPermission(
                requireContext(), Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            Toast.makeText(requireContext(), getString(R.string.stt_permission), Toast.LENGTH_SHORT).show()
            return
        }
        if (isListening) return
        isListening = true
        btnMic.isSelected = true
        setState(ListenState.STARTING)

        val mainActivity = activity as? MainActivity ?: return
        val langTag = mainActivity.selectedLangTag   // e.g. "hi-IN", "zh-CN"

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, langTag)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, langTag)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        }
        speechRecognizer?.startListening(intent)
    }

    private fun stopListening() {
        if (!isListening) return
        speechRecognizer?.stopListening()
    }

    // ── Query handling ─────────────────────────────────────────────────────
    private fun handleQuery(spokenText: String) {
        val mainActivity = activity as? MainActivity ?: return
        val langBase = mainActivity.selectedLang.language   // e.g. "hi", "zh"
        val prefix   = langPrefixes[langBase] ?: ""
        val backendQuery = "$prefix$spokenText"

        // Show user's spoken words (not the prefixed version)
        adapter.addMessage(ChatMessage(spokenText, isUser = true))
        rvChat.smoothScrollToPosition(messages.size - 1)
        setState(ListenState.FETCHING)

        val url     = "${mainActivity.baseUrl}/ask"
        val payload = gson.toJson(mapOf("query" to backendQuery))
        val reqBody = payload.toRequestBody("application/json".toMediaType())
        val req     = Request.Builder().url(url).post(reqBody).build()
        val t0      = System.currentTimeMillis()

        client.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("AssistantFragment", "Failed in ${System.currentTimeMillis()-t0}ms: ${e.message}")
                scope.launch {
                    setState(ListenState.IDLE)
                    Toast.makeText(requireContext(), "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
            override fun onResponse(call: Call, response: Response) {
                val latency = System.currentTimeMillis() - t0
                response.use {
                    val body   = it.body?.string() ?: "{}"
                    val json   = gson.fromJson(body, Map::class.java)
                    val answer = json["answer"]?.toString()
                        ?: getString(R.string.assistant_fallback_answer)
                    Log.i("AssistantFragment",
                        "lang=$langBase latency=${latency}ms answer=\"$answer\"")
                    scope.launch {
                        adapter.addMessage(ChatMessage(answer, isUser = false))
                        rvChat.smoothScrollToPosition(messages.size - 1)
                        setState(ListenState.IDLE)
                        mainActivity.speakText(answer)
                    }
                }
            }
        })
    }

    // ── UI helpers ─────────────────────────────────────────────────────────
    private enum class ListenState { IDLE, STARTING, LISTENING, PROCESSING, FETCHING }

    private fun setState(s: ListenState) {
        tvListeningState.text = when (s) {
            ListenState.IDLE       -> getString(R.string.hint_hold_to_talk)
            ListenState.STARTING   -> getString(R.string.state_starting)
            ListenState.LISTENING  -> getString(R.string.state_listening)
            ListenState.PROCESSING -> getString(R.string.state_processing)
            ListenState.FETCHING   -> getString(R.string.state_fetching)
        }
    }

    /** Shows the BCP-47 language code abbreviation, e.g. "EN", "HI", "ZH". */
    private fun updateLanguageBadge() {
        val mainActivity = activity as? MainActivity ?: return
        tvLanguageBadge.text = mainActivity.selectedLang.language.uppercase()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        speechRecognizer?.destroy()
        speechRecognizer = null
        scope.cancel()
    }
}
