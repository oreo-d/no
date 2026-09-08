package com.example.elijah

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Small persistent vector store for on-device RAG.
 *
 * It keeps document chunks and their embeddings in SharedPreferences.
 * For a larger corpus, replace this implementation with Room/SQLite,
 * but keep the same interface.
 */
class VectorStore(
    context: Context,
    private val embeddingModel: EmbeddingModel
) {
    data class Document(
        val id: String,
        val text: String,
        val source: String = "local",
        val metadata: Map<String, String> = emptyMap()
    )

    data class SearchResult(
        val document: Document,
        val score: Float
    )

    private val prefs = context.getSharedPreferences("elijah_vector_store_v1", Context.MODE_PRIVATE)
    private val docs = LinkedHashMap<String, Document>()
    private val vectors = HashMap<String, FloatArray>()

    init {
        load()
    }

    @Synchronized
    fun upsert(document: Document) {
        docs[document.id] = document
        vectors[document.id] = embeddingModel.embed(document.text)
        save()
    }

    @Synchronized
    fun upsertAll(documents: List<Document>) {
        documents.forEach {
            docs[it.id] = it
            vectors[it.id] = embeddingModel.embed(it.text)
        }
        save()
    }

    @Synchronized
    fun remove(id: String) {
        docs.remove(id)
        vectors.remove(id)
        save()
    }

    @Synchronized
    fun clear() {
        docs.clear()
        vectors.clear()
        prefs.edit().clear().apply()
    }

    @Synchronized
    fun size(): Int = docs.size

    @Synchronized
    fun search(query: String, topK: Int = 5, minScore: Float = -1f): List<SearchResult> {
        if (docs.isEmpty()) return emptyList()
        val q = embeddingModel.embed(query)

        return docs.values.mapNotNull { doc ->
            val vector = vectors[doc.id] ?: return@mapNotNull null
            val score = embeddingModel.similarity(q, vector)
            if (score >= minScore) SearchResult(doc, score) else null
        }.sortedByDescending { it.score }
            .take(topK.coerceIn(1, 20))
    }

    private fun save() {
        val array = JSONArray()
        docs.values.forEach { doc ->
            val item = JSONObject()
            item.put("id", doc.id)
            item.put("text", doc.text)
            item.put("source", doc.source)
            item.put("metadata", JSONObject(doc.metadata))
            array.put(item)
        }
        prefs.edit().putString("documents", array.toString()).apply()
    }

    private fun load() {
        val raw = prefs.getString("documents", null) ?: return
        runCatching {
            val array = JSONArray(raw)
            for (i in 0 until array.length()) {
                val item = array.getJSONObject(i)
                val metadataJson = item.optJSONObject("metadata") ?: JSONObject()
                val metadata = mutableMapOf<String, String>()
                metadataJson.keys().forEach { key ->
                    metadata[key] = metadataJson.optString(key)
                }

                val doc = Document(
                    id = item.getString("id"),
                    text = item.getString("text"),
                    source = item.optString("source", "local"),
                    metadata = metadata
                )
                docs[doc.id] = doc
                vectors[doc.id] = embeddingModel.embed(doc.text)
            }
        }.onFailure {
            docs.clear()
            vectors.clear()
        }
    }
}
