package com.example.safedrivemonitor

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.gson.Gson
import kotlinx.coroutines.*
import okhttp3.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * ReportsFragment
 * ---------------
 * Fetches recent trip events from /events and displays them.
 * Lets the driver download or open the PDF trip report.
 * (No changes to backend — same endpoints as before.)
 */
class ReportsFragment : Fragment() {

    private lateinit var tvStatus: TextView
    private lateinit var btnDownload: Button
    private lateinit var btnOpen: Button
    private lateinit var rvEvents: RecyclerView
    private lateinit var eventAdapter: EventAdapter

    private val client = OkHttpClient()
    private val gson   = Gson()
    private val scope  = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var lastFile: File? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_reports, container, false)

        tvStatus    = view.findViewById(R.id.tvStatus)
        btnDownload = view.findViewById(R.id.btnDownload)
        btnOpen     = view.findViewById(R.id.btnOpen)
        rvEvents    = view.findViewById(R.id.rvEvents)

        eventAdapter = EventAdapter(mutableListOf())
        rvEvents.layoutManager = LinearLayoutManager(requireContext())
        rvEvents.adapter = eventAdapter

        btnDownload.setOnClickListener { downloadReport() }
        btnOpen.setOnClickListener     { openReport() }

        fetchEvents()

        return view
    }

    private fun fetchEvents() {
        val mainActivity = activity as? MainActivity ?: return
        val url = "${mainActivity.baseUrl}/events"
        tvStatus.text = "Fetching recent events…"

        client.newCall(Request.Builder().url(url).get().build())
            .enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    scope.launch {
                        tvStatus.text = "Failed to fetch events: ${e.message}"
                    }
                }
                override fun onResponse(call: Call, response: Response) {
                    response.use {
                        if (!it.isSuccessful) {
                            scope.launch { tvStatus.text = "Failed: ${it.message}" }
                            return
                        }
                        val body   = it.body?.string() ?: "[]"
                        val events = gson.fromJson(body, Array<String>::class.java)
                        scope.launch {
                            eventAdapter.clearAll()
                            events.reversed().forEach { e -> eventAdapter.addEvent(e) }
                            tvStatus.text = "${events.size} events loaded"
                        }
                    }
                }
            })
    }

    private fun downloadReport() {
        val mainActivity = activity as? MainActivity ?: return
        val url = "${mainActivity.baseUrl}/report"
        tvStatus.text = "Downloading report…"

        client.newCall(Request.Builder().url(url).get().build())
            .enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    scope.launch {
                        tvStatus.text = "Download failed: ${e.message}"
                        Toast.makeText(requireContext(), "Download failed", Toast.LENGTH_SHORT).show()
                    }
                }
                override fun onResponse(call: Call, response: Response) {
                    response.use {
                        if (!it.isSuccessful) {
                            scope.launch { tvStatus.text = "Failed: ${it.message}" }
                            return
                        }
                        val bytes = it.body?.bytes() ?: return
                        val file  = File(requireContext().getExternalFilesDir(null), "trip_report.pdf")
                        FileOutputStream(file).use { fos -> fos.write(bytes) }
                        lastFile = file
                        scope.launch {
                            tvStatus.text = "Saved: ${file.name}"
                            Toast.makeText(requireContext(), "Report downloaded", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            })
    }

    private fun openReport() {
        val file = lastFile ?: run {
            Toast.makeText(requireContext(), "No report downloaded yet", Toast.LENGTH_SHORT).show()
            return
        }
        val uri = FileProvider.getUriForFile(
            requireContext(), "${requireContext().packageName}.provider", file
        )
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/pdf")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try { startActivity(intent) }
        catch (e: Exception) {
            Toast.makeText(requireContext(), "No PDF viewer installed", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        scope.cancel()
    }
}
