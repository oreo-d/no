package com.example.elijah

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

/**
 * OpenAI-compatible realtime chat client. Works with hosted APIs or a local
 * server exposing an OpenAI-compatible /v1/chat/completions endpoint.
 * No provider SDK is required.
 */
class RealtimeLLM(
    private val endpoint: String,
    private val model: String,
    private val apiKey: String? = null,
    private val temperature: Double = 0.2,
    private val timeoutMs: Int = 60_000
) : LocalLLM {

    override fun generate(prompt: String, maxTokens: Int): String =
        complete(buildMessages(prompt), maxTokens)

    fun complete(messages: List<Message>, maxTokens: Int = 512): String {
        val connection = openConnection(stream = false)
        val body = JSONObject().apply {
            put("model", model)
            put("messages", JSONArray().apply {
                messages.forEach { put(JSONObject().apply {
                    put("role", it.role)
                    put("content", it.content)
                }) }
            })
            put("temperature", temperature)
            put("max_tokens", maxTokens)
            put("stream", false)
        }
        writeBody(connection, body.toString())
        val response = readResponse(connection)
        return parseAnswer(response)
    }

    /** Streams OpenAI-compatible SSE data. Callback receives incremental text. */
    fun stream(messages: List<Message>, maxTokens: Int = 512, onToken: (String) -> Unit): String {
        val connection = openConnection(stream = true)
        val body = JSONObject().apply {
            put("model", model)
            put("messages", JSONArray().apply {
                messages.forEach { put(JSONObject().apply {
                    put("role", it.role)
                    put("content", it.content)
                }) }
            })
            put("temperature", temperature)
            put("max_tokens", maxTokens)
            put("stream", true)
        }
        writeBody(connection, body.toString())

        val output = StringBuilder()
        BufferedReader(InputStreamReader(connection.inputStream, StandardCharsets.UTF_8)).use { reader ->
            while (true) {
                val line = reader.readLine() ?: break
                if (!line.startsWith("data:")) continue
                val data = line.removePrefix("data:").trim()
                if (data == "[DONE]") break
                val token = runCatching {
                    JSONObject(data).optJSONArray("choices")?.optJSONObject(0)
                        ?.optJSONObject("delta")?.optString("content", "") ?: ""
                }.getOrDefault("")
                if (token.isNotEmpty()) {
                    output.append(token)
                    onToken(token)
                }
            }
        }
        connection.disconnect()
        return output.toString()
    }

    data class Message(val role: String, val content: String)

    private fun buildMessages(prompt: String): List<Message> = listOf(
        Message("system", "You are Elijah AI. Answer accurately and concisely. Use the supplied RAG evidence and do not invent unsupported facts."),
        Message("user", prompt)
    )

    private fun openConnection(stream: Boolean): HttpURLConnection {
        val url = URL(endpoint.trimEnd('/'))
        return (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = timeoutMs
            readTimeout = timeoutMs
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Accept", if (stream) "text/event-stream" else "application/json")
            if (!apiKey.isNullOrBlank()) setRequestProperty("Authorization", "Bearer $apiKey")
        }
    }

    private fun writeBody(connection: HttpURLConnection, body: String) {
        connection.outputStream.use { it.write(body.toByteArray(StandardCharsets.UTF_8)) }
        if (connection.responseCode !in 200..299) {
            val error = runCatching { connection.errorStream?.bufferedReader()?.readText() ?: "HTTP ${connection.responseCode}" }
                .getOrDefault("HTTP ${connection.responseCode}")
            connection.disconnect()
            throw IllegalStateException("LLM request failed: $error")
        }
    }

    private fun readResponse(connection: HttpURLConnection): String {
        val text = connection.inputStream.bufferedReader().use { it.readText() }
        connection.disconnect()
        return text
    }

    private fun parseAnswer(raw: String): String {
        val json = JSONObject(raw)
        return json.optJSONArray("choices")?.optJSONObject(0)
            ?.optJSONObject("message")?.optString("content", "")?.trim()
            ?: throw IllegalStateException("LLM returned no message content")
    }
}
