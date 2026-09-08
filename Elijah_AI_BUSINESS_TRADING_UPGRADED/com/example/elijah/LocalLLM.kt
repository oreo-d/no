package com.example.elijah

/**
 * Local LLM abstraction. Put a real TFLite/MediaPipe/GGUF-backed model behind
 * this interface. The included generator is an offline safe fallback, so the
 * RAG app remains functional before a model asset is installed.
 */
interface LocalLLM {
    fun generate(prompt:String, maxTokens:Int=256):String
}

class OfflineLocalLLM : LocalLLM {
    override fun generate(prompt:String,maxTokens:Int):String {
        val answer=prompt.substringAfter("QUESTION:",prompt).substringBefore("ANSWER:").trim()
        return if(answer.isBlank()) "I need a question to answer." else
            "I can answer locally from the retrieved context. For a full generative response, attach a compatible on-device LLM to LocalLLM."
    }
}


/** Bridges a LocalLLM into the RAG generator contract. */
class LocalLLMRagGenerator(private val llm: LocalLLM) : RagPipeline.Generator {
    override fun generate(question: String, context: List<RagPipeline.Chunk>): String {
        val evidence = context.joinToString("\n\n") {
            "[${it.source}] ${it.text}"
        }
        val prompt = """
            You are Elijah AI, an offline assistant.
            Use the supplied evidence when answering. Do not invent facts that are not supported.
            QUESTION:
            $question
            EVIDENCE:
            $evidence
            ANSWER:
        """.trimIndent()
        return llm.generate(prompt, 256)
    }
}
