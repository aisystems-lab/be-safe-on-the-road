package com.example.safedrivemonitor

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

/**
 * EventAdapter
 * ------------
 * Displays trip events (strings) in ReportsFragment's RecyclerView.
 */
class EventAdapter(private val events: MutableList<String>) :
    RecyclerView.Adapter<EventAdapter.EventViewHolder>() {

    inner class EventViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvLine: TextView    = view.findViewById(R.id.tvLine)
        val tvIndex: TextView   = view.findViewById(R.id.tvIndex)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): EventViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_event, parent, false)
        return EventViewHolder(view)
    }

    override fun onBindViewHolder(holder: EventViewHolder, position: Int) {
        holder.tvLine.text  = events[position]
        holder.tvIndex.text = "#${events.size - position}"
    }

    override fun getItemCount() = events.size

    fun addEvent(event: String) {
        events.add(event)
        notifyItemInserted(events.size - 1)
    }

    fun clearAll() {
        events.clear()
        notifyDataSetChanged()
    }
}
