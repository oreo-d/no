package com.example.elijah

/**
 * End-to-end realtime RAG service:
 * query -> vector retrieval -> grounded prompt -> streamed LLM answer.
 */
class RealtimeRagService(
    private val vectorStore: VectorStore,
    private val llm: RealtimeLLM
) {
    data class Result(val text: String, val sources: List<RagPipeline.Chunk>)

    fun streamAnswer(
        question: String,
        topK: Int = 5,
        minScore: Float = 0.12f,
        onToken: (String) -> Unit
    ): Result {
        val sources = vectorStore.search(question, topK, minScore).map {
            RagPipeline.Chunk(it.document.id, it.document.text, it.document.source, it.score)
        }

        val evidence = if (sources.isEmpty()) {
            "No matching local knowledge was retrieved. Clearly state uncertainty when the answer is not known."
        } else {
            sources.joinToString("\n\n") {
                "SOURCE: ${it.source}\nCONTENT: ${it.text}"
            }
        }

        val system = """
            You are Elijah AI, a helpful realtime assistant.
            Ground factual answers in the evidence below. If evidence is insufficient,
            say so instead of inventing facts. You may use general reasoning, but do not
            claim that unsupported details came from the knowledge base.
            Keep answers clear and useful.
        """.trimIndent()
        val user = "QUESTION:\n$question\n\nEVIDENCE:\n$evidence\n\nAnswer the question directly."

        val text = llm.stream(
            listOf(
                RealtimeLLM.Message("system", system),
                RealtimeLLM.Message("user", user)
            ),
            maxTokens = 700,
            onToken = onToken
        )
        return Result(text, sources)
    }
}
