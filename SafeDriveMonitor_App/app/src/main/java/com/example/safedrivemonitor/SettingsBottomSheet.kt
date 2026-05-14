package com.example.safedrivemonitor

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.switchmaterial.SwitchMaterial
import com.google.android.material.textfield.TextInputEditText

/**
 * SettingsBottomSheet – supports 9 languages, light/dark mode toggle.
 */
class SettingsBottomSheet : BottomSheetDialogFragment() {

    // Ordered list of (display label → BCP-47 language tag)
    private val languages = listOf(
        "English"            to "en-US",
        "Spanish / Español"  to "es-ES",
        "Hindi / हिन्दी"      to "hi-IN",
        "French / Français"  to "fr-FR",
        "Marathi / मराठी"    to "mr-IN",
        "Japanese / 日本語"   to "ja-JP",
        "German / Deutsch"   to "de-DE",
        "Chinese / 中文"      to "zh-CN",
    )

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.bottomsheet_settings, container, false)

        val etServer   = view.findViewById<TextInputEditText>(R.id.etServer)
        val spLang     = view.findViewById<Spinner>(R.id.spLang)
        val spGender   = view.findViewById<Spinner>(R.id.spGender)
        val swDark     = view.findViewById<SwitchMaterial>(R.id.swDarkMode)
        val btnApply   = view.findViewById<MaterialButton>(R.id.btnApply)

        val mainActivity = activity as? MainActivity ?: return view

        // ── Pre-populate ───────────────────────────────────────────────
        etServer.setText(mainActivity.baseUrl)

        // Language spinner
        val langLabels = languages.map { it.first }
        spLang.adapter = buildAdapter(langLabels)
        val currentTag = mainActivity.selectedLangTag
        val currentIdx = languages.indexOfFirst { it.second.startsWith(currentTag.substringBefore('-'), true) }
        spLang.setSelection(if (currentIdx >= 0) currentIdx else 0)

        // Gender spinner
        val genderOptions = listOf(
            getString(R.string.voice_female),
            getString(R.string.voice_male)
        )
        spGender.adapter = buildAdapter(genderOptions)
        spGender.setSelection(genderOptions.indexOf(mainActivity.selectedGender))

        // Dark mode switch
        swDark.isChecked = mainActivity.isDarkMode

        // ── Apply ──────────────────────────────────────────────────────
        btnApply.setOnClickListener {
            val newUrl    = etServer.text.toString().trim()
            val langTag   = languages[spLang.selectedItemPosition].second
            val gender    = spGender.selectedItem.toString()
            val darkMode  = swDark.isChecked

            mainActivity.applySettings(newUrl, langTag, gender, darkMode)

            Toast.makeText(requireContext(), getString(R.string.settings_saved), Toast.LENGTH_SHORT).show()
            dismiss()
        }

        return view
    }

    private fun buildAdapter(items: List<String>): ArrayAdapter<String> {
        val adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, items)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        return adapter
    }
}
