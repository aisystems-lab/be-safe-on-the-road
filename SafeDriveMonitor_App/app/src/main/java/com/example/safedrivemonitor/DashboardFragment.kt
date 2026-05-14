//package com.example.safedrivemonitor
//
//import android.os.Bundle
//import android.view.LayoutInflater
//import android.view.View
//import android.view.ViewGroup
//import android.widget.*
//import androidx.fragment.app.Fragment
//import com.google.android.material.progressindicator.CircularProgressIndicator
//import kotlinx.coroutines.*
//import okhttp3.*
//import com.google.gson.Gson
//import java.io.IOException
//import okhttp3.MediaType.Companion.toMediaType
//import okhttp3.RequestBody.Companion.toRequestBody
//
///**
// * DashboardFragment
// * -----------------
// * Shows the current risk level on an animated ring, lets the driver
// * select a risk level with a slider, and sends a risk alert to the
// * backend which speaks the result via TTS.
// */
//class DashboardFragment : Fragment() {
//
//    private lateinit var ring: CircularProgressIndicator
//    private lateinit var tvStatus: TextView
//    private lateinit var tvRiskLine: TextView
//    private lateinit var tvRiskLevel: TextView
//    private lateinit var riskBar: SeekBar
//    private lateinit var btnSpeakRisk: com.google.android.material.button.MaterialButton
//
//    private val client = OkHttpClient()
//    private val gson = Gson()
//    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
//
//    override fun onCreateView(
//        inflater: LayoutInflater, container: ViewGroup?,
//        savedInstanceState: Bundle?
//    ): View {
//        val view = inflater.inflate(R.layout.fragment_dashboard, container, false)
//
//        ring         = view.findViewById(R.id.ring)
//        tvStatus     = view.findViewById(R.id.tvStatus)
//        tvRiskLine   = view.findViewById(R.id.tvRiskLine)
//        tvRiskLevel  = view.findViewById(R.id.tvRiskLevel)
//        riskBar      = view.findViewById(R.id.riskBar)
//        btnSpeakRisk = view.findViewById(R.id.btnSpeakRisk)
//
//        updateUI(0)
//
//        riskBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
//            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
//                updateUI(progress)
//            }
//            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
//            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
//        })
//
//        btnSpeakRisk.setOnClickListener {
//            sendRiskAlert(riskBar.progress)
//        }
//
//        return view
//    }
//
//    private fun updateUI(level: Int) {
//        val statusText: String
//        val color: Int
//        val percent: Int
//        val desc: String
//
//        when (level) {
//            0 -> {
//                statusText = "SAFE"
//                color      = resources.getColor(R.color.risk_safe, null)
//                percent    = 10
//                desc       = "All clear. Drive comfortably."
//            }
//            1 -> {
//                statusText = "CAUTION"
//                color      = resources.getColor(R.color.risk_caution, null)
//                percent    = 30
//                desc       = "Minor alert. Stay focused."
//            }
//            2 -> {
//                statusText = "MODERATE"
//                color      = resources.getColor(R.color.risk_moderate, null)
//                percent    = 55
//                desc       = "Stay alert and slow down."
//            }
//            3 -> {
//                statusText = "HIGH"
//                color      = resources.getColor(R.color.risk_high, null)
//                percent    = 80
//                desc       = "High risk. Reduce speed now."
//            }
//            else -> {
//                statusText = "CRITICAL"
//                color      = resources.getColor(R.color.risk_critical, null)
//                percent    = 100
//                desc       = "Critical! Stop safely immediately."
//            }
//        }
//
//        ring.setIndicatorColor(color)
//        ring.progress  = percent
//        tvStatus.text  = statusText
//        tvStatus.setTextColor(color)
//        tvRiskLine.text = desc
//        tvRiskLevel.text = "Level $level / 4"
//    }
//
//    private fun sendRiskAlert(level: Int) {
//        val mainActivity = activity as? MainActivity ?: return
//        val url     = "${mainActivity.baseUrl}/risk_alert"
//        val payload = gson.toJson(mapOf("risk_level" to level))
//        val reqBody = payload.toRequestBody("application/json".toMediaType())
//        val req     = Request.Builder().url(url).post(reqBody).build()
//
//        client.newCall(req).enqueue(object : Callback {
//            override fun onFailure(call: Call, e: IOException) {
//                scope.launch {
//                    Toast.makeText(requireContext(), "Error: ${e.message}", Toast.LENGTH_SHORT).show()
//                }
//            }
//            override fun onResponse(call: Call, response: Response) {
//                response.use {
//                    val body = it.body?.string() ?: "{}"
//                    val msg  = gson.fromJson(body, Map::class.java)["message"]?.toString() ?: "No message"
//                    scope.launch {
//                        mainActivity.speakText(msg)
//                        Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
//                    }
//                }
//            }
//        })
//    }
//
//    override fun onDestroyView() {
//        super.onDestroyView()
//        scope.cancel()
//    }
//}

package com.example.safedrivemonitor

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import com.google.android.material.progressindicator.CircularProgressIndicator
import kotlinx.coroutines.*
import okhttp3.*
import com.google.gson.Gson
import java.io.IOException
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody

class DashboardFragment : Fragment() {

    // ── Views ──────────────────────────────────────────────────────────────
    private lateinit var ring: CircularProgressIndicator
    private lateinit var tvStatus: TextView
    private lateinit var tvLevel: TextView
    private lateinit var tvRiskLine: TextView
    private lateinit var riskBar: SeekBar
    private lateinit var btnSpeakRisk: com.google.android.material.button.MaterialButton
    private lateinit var historyRow: LinearLayout
    private lateinit var riskButtonsRow: LinearLayout

    // ── State ──────────────────────────────────────────────────────────────
    private val history = ArrayDeque<Int>(6).also { it.add(0) }   // seed with SAFE
    private val riskButtons = mutableListOf<View>()

    private val client = OkHttpClient()
    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // ── Risk data ──────────────────────────────────────────────────────────
    data class RiskInfo(
        val label: String,
        val colorHex: String,
        val percent: Int,
        val desc: String,
        val sym: String
    )

    private val RISK = listOf(
        RiskInfo("SAFE",     "#009E73", 10,  "All clear. Drive comfortably.",        "✓"),
        RiskInfo("CAUTION",  "#56B4E9", 30,  "Minor alert. Stay focused.",           "!"),
        RiskInfo("MODERATE", "#E69F00", 55,  "Stay alert and slow down.",            "⚠"),
        RiskInfo("HIGH",     "#D55E00", 80,  "High risk. Reduce speed now.",         "▲"),
        RiskInfo("CRITICAL", "#CC79A7", 100, "Critical! Stop safely immediately.",   "✕")
    )

    // ── Lifecycle ──────────────────────────────────────────────────────────
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_dashboard, container, false)

        ring           = view.findViewById(R.id.ring)
        tvStatus       = view.findViewById(R.id.tvStatus)
        tvLevel        = view.findViewById(R.id.tvLevel)
        tvRiskLine     = view.findViewById(R.id.tvRiskLine)
        riskBar        = view.findViewById(R.id.riskBar)
        btnSpeakRisk   = view.findViewById(R.id.btnSpeakRisk)
        historyRow     = view.findViewById(R.id.historyRow)
        riskButtonsRow = view.findViewById(R.id.riskButtonsRow)

        buildRiskButtons()
        updateUI(0)

        riskBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) applyLevel(progress)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        btnSpeakRisk.setOnClickListener { sendRiskAlert(riskBar.progress) }

        return view
    }

    override fun onDestroyView() {
        super.onDestroyView()
        scope.cancel()
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /** Called from slider or buttons; pushes to history then redraws. */
    private fun applyLevel(level: Int) {
        if (history.size >= 6) history.removeFirst()
        history.addLast(level)
        updateUI(level)
    }

    /** Rebuild the full UI for the given risk level. */
    private fun updateUI(level: Int) {
        val r = RISK[level]
        val color = Color.parseColor(r.colorHex)

        // Ring
        ring.setIndicatorColor(color)
        ring.setProgressCompat(r.percent, true)

        // Labels
        tvStatus.text = r.label
        tvStatus.setTextColor(color)
        tvLevel.text  = "Level $level / 4"
        tvRiskLine.text = r.desc

        // Slider (no listener re-trigger needed; we gate on fromUser)
        riskBar.progress = level

        // History dots
        redrawHistoryDots()

        // Quick-select buttons highlight
        riskButtons.forEachIndexed { i, btn ->
            val alpha = if (i == level) 1f else 0.3f
            btn.alpha = alpha
            (btn.background as? GradientDrawable)?.let { bg ->
                if (i == level) {
                    bg.setStroke(dpToPx(2), Color.WHITE)
                } else {
                    bg.setStroke(dpToPx(2), Color.TRANSPARENT)
                }
            }
        }
    }

    /** Build the 5 quick-select colour buttons once. */
    private fun buildRiskButtons() {
        riskButtonsRow.removeAllViews()
        riskButtons.clear()

        RISK.forEachIndexed { i, r ->
            val color = Color.parseColor(r.colorHex)

            // Circular drawable background
            val bg = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(6).toFloat()
                setColor(color)
                setStroke(dpToPx(2), Color.TRANSPARENT)
            }

            val btn = TextView(requireContext()).apply {
                text = r.sym
                textSize = 13f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                background = bg
                alpha = 0.3f
                isClickable = true
                isFocusable = true
                setOnClickListener { applyLevel(i) }
            }

            val params = LinearLayout.LayoutParams(0, dpToPx(32), 1f).apply {
                marginEnd = if (i < 4) dpToPx(5) else 0
            }
            riskButtonsRow.addView(btn, params)
            riskButtons.add(btn)
        }
    }

    /** Rebuild the history dot row from the current history deque. */
    private fun redrawHistoryDots() {
        // Keep the "HISTORY" label (first child), remove everything after it
        while (historyRow.childCount > 1) historyRow.removeViewAt(1)

        history.forEachIndexed { i, lvl ->
            val isLast = i == history.size - 1
            val color = Color.parseColor(RISK[lvl].colorHex)
            val dotSize = if (isLast) dpToPx(26) else dpToPx(22)

            // Connector line (before each dot except first)
            if (i > 0) {
                val line = View(requireContext()).apply {
                    setBackgroundColor(Color.parseColor("#1A2A3D"))
                }
                val lp = LinearLayout.LayoutParams(0, dpToPx(2), 1f).apply {
                    gravity = Gravity.CENTER_VERTICAL
                }
                historyRow.addView(line, lp)
            }

            // Dot
            val dotBg = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(color)
                if (isLast) setStroke(dpToPx(2), Color.parseColor("#80FFFFFF"))
                else alpha = (0.55 * 255).toInt()
            }

            val dot = TextView(requireContext()).apply {
                text = RISK[lvl].sym
                textSize = 9f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                background = dotBg
                alpha = if (isLast) 1f else 0.65f
            }

            val dp = LinearLayout.LayoutParams(dotSize, dotSize).apply {
                gravity = Gravity.CENTER_VERTICAL
            }
            historyRow.addView(dot, dp)
        }
    }

    /** Send alert to backend and speak the response. */
    private fun sendRiskAlert(level: Int) {
        val mainActivity = activity as? MainActivity ?: return
        val url = "${mainActivity.baseUrl}/risk_alert"

        val payload = gson.toJson(mapOf("risk_level" to level))
        val reqBody = payload.toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url(url).post(reqBody).build()

        client.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                scope.launch {
                    Toast.makeText(requireContext(), "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
            override fun onResponse(call: Call, response: Response) {
                response.use {
                    val body = it.body?.string() ?: "{}"
                    val msg = gson.fromJson(body, Map::class.java)["message"]?.toString()
                        ?: "No message"
                    scope.launch {
                        Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
                        mainActivity.speakText(msg)
                    }
                }
            }
        })
    }

    private fun dpToPx(dp: Int): Int =
        (dp * resources.displayMetrics.density).toInt()
}
