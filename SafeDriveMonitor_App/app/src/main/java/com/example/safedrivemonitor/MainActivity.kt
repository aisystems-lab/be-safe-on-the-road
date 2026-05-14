package com.example.safedrivemonitor

import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import android.view.Menu
import android.view.MenuItem
import java.util.Locale

/**
 * MainActivity – hosts Dashboard / Assistant / Report tabs.
 *
 * New in v2:
 *  • 9-language support via BCP-47 tags stored in SharedPreferences
 *  • Light / Dark mode persisted and applied at launch
 */
class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var tts: TextToSpeech
    private var ttsReady = false
    private var selectedVoice: Voice? = null

    // ── SharedPreferences-backed settings ──────────────────────────────────

    var baseUrl: String
        get() = prefs.getString("baseUrl", "http://192.168.1.63:8000")!!
        set(v) = prefs.edit().putString("baseUrl", v).apply()

    /**
     * BCP-47 language tag, e.g. "en-US", "hi-IN", "zh-CN".
     * Used for both SpeechRecognizer locale and TTS locale.
     */
    var selectedLangTag: String
        get() = prefs.getString("langTag", "en-US")!!
        private set(v) = prefs.edit().putString("langTag", v).apply()

    /** Convenience Locale built from [selectedLangTag]. */
    val selectedLang: Locale
        get() = Locale.forLanguageTag(selectedLangTag)

    var selectedGender: String
        get() = prefs.getString("gender", "Female") ?: "Female"
        private set(v) = prefs.edit().putString("gender", v).apply()

    var isDarkMode: Boolean
        get() = prefs.getBoolean("darkMode", true)   // default dark
        private set(v) = prefs.edit().putBoolean("darkMode", v).apply()

    private val prefs get() = getSharedPreferences("app", MODE_PRIVATE)

    // ── Lifecycle ──────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        // Apply saved theme BEFORE super/setContentView
        applyThemeMode(isDarkMode)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tts = TextToSpeech(this, this)

        val toolbar = findViewById<MaterialToolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)

        val pager = findViewById<ViewPager2>(R.id.pager)
        pager.adapter = object : FragmentStateAdapter(supportFragmentManager, lifecycle) {
            override fun getItemCount() = 3
            override fun createFragment(pos: Int) = when (pos) {
                0    -> DashboardFragment()
                1    -> AssistantFragment()
                else -> ReportsFragment()
            }
        }

        val tabs = findViewById<TabLayout>(R.id.tabs)
        TabLayoutMediator(tabs, pager) { tab, pos ->
            tab.text = when (pos) {
                0    -> getString(R.string.tab_dashboard)
                1    -> getString(R.string.tab_assistant)
                else -> getString(R.string.tab_report)
            }
        }.attach()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == R.id.action_settings) {
            SettingsBottomSheet().show(supportFragmentManager, "settings")
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onInit(status: Int) {
        ttsReady = status == TextToSpeech.SUCCESS
        if (ttsReady) {
            tts.language = selectedLang
            applyVoiceGender(selectedGender)
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Called by SettingsBottomSheet when the user taps Save.
     * @param langTag  BCP-47 tag e.g. "hi-IN", "ja-JP", "zh-CN"
     * @param darkMode true = dark theme
     */
    fun applySettings(newUrl: String, langTag: String, gender: String, darkMode: Boolean) {
        baseUrl         = newUrl
        selectedLangTag = langTag
        selectedGender  = gender
        isDarkMode      = darkMode

        if (ttsReady) {
            tts.language = selectedLang
            applyVoiceGender(gender)
        }

        // Recreate the activity so the theme change takes effect immediately
        if (darkMode != (AppCompatDelegate.getDefaultNightMode() == AppCompatDelegate.MODE_NIGHT_YES)) {
            applyThemeMode(darkMode)
            recreate()
        }
    }

    fun speakText(text: String) {
        if (!ttsReady) return
        tts.language = selectedLang
        selectedVoice?.let { tts.voice = it }
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "utt")
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private fun applyThemeMode(dark: Boolean) {
        AppCompatDelegate.setDefaultNightMode(
            if (dark) AppCompatDelegate.MODE_NIGHT_YES
            else      AppCompatDelegate.MODE_NIGHT_NO
        )
    }

    private fun applyVoiceGender(gender: String) {
        if (!ttsReady) return
        val voices = tts.voices?.filter {
            it.locale.language == selectedLang.language
        } ?: emptyList()
        selectedVoice = if (gender == "Male")
            voices.firstOrNull { it.name.contains("male", ignoreCase = true) }
        else
            voices.firstOrNull { it.name.contains("female", ignoreCase = true) }
        selectedVoice?.let { tts.voice = it }
    }

    override fun onDestroy() {
        tts.shutdown()
        super.onDestroy()
    }
}
