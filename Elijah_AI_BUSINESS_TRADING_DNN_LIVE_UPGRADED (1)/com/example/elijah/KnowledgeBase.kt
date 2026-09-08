package com.example.elijah

/**
 * Default local knowledge used to demonstrate RAG immediately.
 * Replace/add documents with app-specific manuals, FAQs, notes, or files.
 */
object KnowledgeBase {

    fun defaultDocuments(): List<VectorStore.Document> = listOf(
        VectorStore.Document(
            id = "kotlin_coroutines",
            source = "Kotlin Guide",
            text = """
                Kotlin coroutines make asynchronous programming easier to read and maintain.
                A coroutine can suspend without blocking the underlying thread.
                viewModelScope is commonly used in Android ViewModels so work is cancelled
                when the ViewModel is cleared. Dispatchers.Default is useful for CPU-heavy work,
                while Dispatchers.IO is intended for blocking input/output operations.
            """.trimIndent()
        ),
        VectorStore.Document(
            id = "android_viewmodel",
            source = "Android Architecture Notes",
            text = """
                An Android ViewModel stores UI-related state and survives configuration changes.
                StateFlow is useful for exposing observable state from a ViewModel.
                Long-running work should be launched from viewModelScope.
                A repository can separate data access from UI logic.
            """.trimIndent()
        ),
        VectorStore.Document(
            id = "rag_basics",
            source = "Elijah AI RAG Notes",
            text = """
                Retrieval-augmented generation combines retrieval with generation.
                The application first embeds a user query, searches a vector index,
                selects relevant document chunks, and then places those chunks into
                the generation context. Retrieval can reduce unsupported answers because
                the generator receives application-specific evidence.
            """.trimIndent()
        ),
        VectorStore.Document(
            id = "embeddings",
            source = "Elijah AI Embedding Notes",
            text = """
                An embedding represents text as a numeric vector. Similar texts should
                produce vectors with high cosine similarity. This project uses a small
                deterministic offline embedding model based on word, phrase, character,
                and positional hashing. It is useful for local retrieval but is not
                equivalent to a pretrained transformer embedding.
            """.trimIndent()
        ),
        VectorStore.Document(
            id = "machine_learning",
            source = "Elijah AI ML Notes",
            text = """
                The local classifier uses a feed-forward neural network with ReLU hidden
                layers and Softmax output. Cross-entropy is used for multi-class learning.
                Adam optimization and gradient clipping improve training stability.
                Corrections can be stored as training examples and replayed later.
            """.trimIndent()
        )
    )
}
