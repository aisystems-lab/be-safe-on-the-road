package com.example.safedrivemonitor

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

/**
 * ChatAdapter
 * -----------
 * Renders user and bot messages in AssistantFragment's RecyclerView.
 * Two view types: TYPE_USER (right-aligned) and TYPE_BOT (left-aligned).
 */
class ChatAdapter(private val messages: MutableList<ChatMessage>) :
    RecyclerView.Adapter<ChatAdapter.MsgViewHolder>() {

    companion object {
        private const val TYPE_USER = 0
        private const val TYPE_BOT  = 1
    }

    inner class MsgViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvMessage: TextView = view.findViewById(R.id.tvMessage)
        val tvSender: TextView  = view.findViewById(R.id.tvSender)
    }

    override fun getItemViewType(position: Int) =
        if (messages[position].isUser) TYPE_USER else TYPE_BOT

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MsgViewHolder {
        val layoutId = if (viewType == TYPE_USER)
            R.layout.item_msg_user else R.layout.item_msg_bot
        val view = LayoutInflater.from(parent.context).inflate(layoutId, parent, false)
        return MsgViewHolder(view)
    }

    override fun onBindViewHolder(holder: MsgViewHolder, position: Int) {
        val msg = messages[position]
        holder.tvMessage.text = msg.text
        holder.tvSender.text  = if (msg.isUser) "You" else "Assistant"
    }

    override fun getItemCount() = messages.size

    fun addMessage(msg: ChatMessage) {
        messages.add(msg)
        notifyItemInserted(messages.size - 1)
    }

    fun clearAll() {
        messages.clear()
        notifyDataSetChanged()
    }
}

/** Simple data model for a single chat message. */
data class ChatMessage(
    val text: String,
    val isUser: Boolean
)
