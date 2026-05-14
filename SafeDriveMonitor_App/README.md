# SafeDriveMonitor

## How to open in Android Studio
1. Unzip the downloaded folder
2. Open **Android Studio → File → Open**
3. Select the `SafeDriveMonitor_Complete` folder
4. Let Gradle sync complete (first sync downloads dependencies)
5. Connect your Android device or start an emulator (API 24+)
6. Click **Run ▶**

---

## Features

### 9-language voice support
Voice queries are recognised in the selected language via Android's built-in
`SpeechRecognizer`. Responses are generated in the same language by prepending
a native-language instruction to the RAG query (no backend changes needed).

Supported: English · Spanish · Hindi · French · Marathi · Japanese · German · Chinese 

### Light / Dark mode
Toggle in Settings sheet. Persisted across app restarts via SharedPreferences.
Uses `AppCompatDelegate` — recreates the Activity when the theme changes.

### Okabe-Ito risk colours (colour-blind safe)
| Level | Colour | Hex |
|-------|--------|-----|
| SAFE | Bluish green | #009E73 |
| CAUTION | Sky blue | #56B4E9 |
| MODERATE | Orange | #E69F00 |
| HIGH | Vermilion | #D55E00 |
| CRITICAL | Reddish purple | #CC79A7 |

Each level also carries a symbol (✓ ! ⚠ ▲ ✕) so the UI is readable without colour.

### Mini risk timeline
A dot-and-line history strip below the risk ring shows the last 6 risk states.

---

## Backend URL
Set your Flask server IP in **Settings → Backend Server**.
Default: `http://192.168.1.63:8000`

Add your IP to `app/src/main/res/xml/network_security_config.xml` if it differs.

---

