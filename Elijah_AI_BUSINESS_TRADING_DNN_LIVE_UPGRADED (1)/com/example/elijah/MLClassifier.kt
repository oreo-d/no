package com.example.elijah

/**
 * Hybrid NLP classifier.
 *
 * It combines:
 *  - lexical/phrase scoring for reliable known intents
 *  - the TextProcessor representation
 *  - learned neural categories handled by NeuralNetwork
 *
 * The rule layer is useful on a completely offline phone because a model
 * with zero training data should still understand common technical words.
 */
class MLClassifier {

    enum class Intent {
        GREETING,
        KOTLIN,
        ANDROID,
        FIREBASE,
        MACHINE_LEARNING,
        PROGRAMMING,
        HELP,
        MATH,
        UNKNOWN
    }

    data class Result(
        val intent: Intent,
        val score: Double,
        val matchedTerms: List<String>
    )

    private val groups = mapOf(
        Intent.GREETING to setOf("hello", "hi", "hey", "good morning", "good evening"),
        Intent.KOTLIN to setOf("kotlin", "kt", "coroutine", "compose", "viewmodel", "jetpack"),
        Intent.ANDROID to setOf("android", "activity", "fragment", "manifest", "apk", "gradle", "avd"),
        Intent.FIREBASE to setOf("firebase", "firestore", "fcm", "storage", "authentication", "realtime database"),
        Intent.MACHINE_LEARNING to setOf(
            "machine learning", "deep learning", "neural network", "nlp",
            "natural language", "backpropagation", "gradient", "training", "model", "ai"
        ),
        Intent.PROGRAMMING to setOf(
            "programming", "code", "coding", "algorithm", "function", "class",
            "variable", "array", "database", "api", "debug", "software"
        ),
        Intent.HELP to setOf("help", "explain", "how do i", "how to", "what does", "teach me"),
        Intent.MATH to setOf("math", "calculate", "equation", "algebra", "derivative", "integral", "sum")
    )

    fun classify(features: FloatArray, message: String): Intent {
        return classifyDetailed(features, message).intent
    }

    fun classifyDetailed(features: FloatArray, message: String): Result {
        val text = message.lowercase()
        val matched = mutableListOf<String>()
        val scores = mutableMapOf<Intent, Double>()

        for ((intent, terms) in groups) {
            var score = 0.0
            for (term in terms) {
                if (text.contains(term)) {
                    // Longer phrases carry more information than short words.
                    score += if (term.contains(" ")) 2.5 else 1.0
                    matched += term
                }
            }
            scores[intent] = score
        }

        // Small text-shape boosts.
        if (text.contains("?")) {
            scores[Intent.HELP] = (scores[Intent.HELP] ?: 0.0) + 0.15
        }

        val best = scores.maxByOrNull { it.value }
        if (best == null || best.value < 0.75) {
            return Result(Intent.UNKNOWN, 0.0, matched.distinct())
        }

        return Result(best.key, best.value, matched.distinct())
    }
}
