package com.example.elijah

/**
 * Retrieval-Augmented Generation orchestration layer.
 *
 * Retrieval is real and local. Generation is deliberately separated behind
 * a Generator interface so the app can use:
 *  - a local LLM,
 *  - an API model,
 *  - or the built-in extractive fallback.
 */
class RagPipeline(
    private val vectorStore: VectorStore,
    private val generator: Generator = ExtractiveGenerator()
) {
    data class Chunk(
        val id: String,
        val text: String,
        val source: String,
        val score: Float
    )

    data class Answer(
        val text: String,
        val sources: List<Chunk>,
        val usedRetrieval: Boolean
    )

    interface Generator {
        fun generate(question: String, context: List<Chunk>): String
    }

    fun answer(
        question: String,
        topK: Int = 4,
        minScore: Float = 0.18f
    ): Answer {
        val results = vectorStore.search(question, topK, minScore)
        if (results.isEmpty()) {
            return Answer(
                text = generator.generate(question, emptyList()),
                sources = emptyList(),
                usedRetrieval = false
            )
        }

        val chunks = results.map {
            Chunk(it.document.id, it.document.text, it.document.source, it.score)
        }

        return Answer(
            text = generator.generate(question, chunks),
            sources = chunks,
            usedRetrieval = true
        )
    }

    /**
     * Simple offline extractive generator. It selects sentences from the
     * retrieved context using token overlap with the question.
     */
    class ExtractiveGenerator : Generator {
        override fun generate(question: String, context: List<Chunk>): String {
            if (context.isEmpty()) {
                return "I don't have a matching local knowledge source yet. " +
                    "Add documents to the knowledge base or connect a generative model."
            }

            val queryTerms = terms(question)
            val candidates = mutableListOf<Pair<String, Int>>()

            context.forEach { chunk ->
                splitSentences(chunk.text).forEach { sentence ->
                    val score = terms(sentence).count { it in queryTerms }
                    if (score > 0) candidates += sentence.trim() to score
                }
            }

            val selected = candidates
                .sortedByDescending { it.second }
                .distinctBy { it.first.lowercase() }
                .take(3)
                .map { it.first }

            val body = if (selected.isEmpty()) {
                context.first().text.take(700)
            } else {
                selected.joinToString(" ")
            }

            return "$body\n\nSources: " +
                context.take(3).joinToString(", ") { it.source }
        }

        private fun terms(text: String): Set<String> =
            text.lowercase()
                .replace(Regex("[^\\p{L}\\p{N}_+#.-]+"), " ")
                .split(Regex("\\s+"))
                .filter { it.length >= 3 }
                .toSet()

        private fun splitSentences(text: String): List<String> =
            text.split(Regex("(?<=[.!?])\\s+"))
                .filter { it.isNotBlank() }
    }
}
