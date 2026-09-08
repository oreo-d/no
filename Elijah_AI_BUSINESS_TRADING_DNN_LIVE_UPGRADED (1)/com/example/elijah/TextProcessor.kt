package com.example.elijah

import kotlin.math.sqrt
import java.util.Locale

/**
 * STRONGER ON-DEVICE NLP PIPELINE
 *
 * This class deliberately uses only Kotlin/Android standard libraries.
 *
 * Pipeline:
 *  1. Unicode/lower-case normalization
 *  2. Tokenization
 *  3. Stop-word filtering
 *  4. Lightweight stemming
 *  5. Word unigrams
 *  6. Word bigrams
 *  7. Character n-grams
 *  8. Text-shape features
 *  9. L2 normalization
 *
 * Hashing keeps the vocabulary memory small, so the app does not need
 * a huge dictionary stored on the phone.
 *
 * IMPORTANT:
 * Feature hashing is not the same thing as a modern transformer/LLM.
 * It is a strong classical NLP representation for a small offline app.
 */
class TextProcessor(
    private val featureSize: Int = 512
) {
    companion object {
        private const val WORD_BUCKETS = 320
        private const val BIGRAM_BUCKETS = 96
        private const val CHAR_BUCKETS = 64
        private const val EXTRA_FEATURES = 32

        private val STOP_WORDS = setOf(
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
            "for", "from", "has", "have", "he", "her", "his", "i", "if",
            "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
            "she", "so", "that", "the", "their", "them", "there", "they",
            "this", "to", "was", "we", "were", "what", "when", "where",
            "which", "who", "why", "will", "with", "you", "your"
        )

        private val POSITIVE = setOf(
            "good", "great", "excellent", "amazing", "love", "like",
            "helpful", "thanks", "thank", "happy", "success", "awesome"
        )

        private val NEGATIVE = setOf(
            "bad", "wrong", "error", "fail", "failed", "broken", "hate",
            "problem", "issue", "bug", "crash", "slow", "confused"
        )
    }

    init {
        require(featureSize >= WORD_BUCKETS + BIGRAM_BUCKETS + CHAR_BUCKETS + EXTRA_FEATURES)
    }

    /**
     * Main NLP -> numerical feature vector.
     */
    fun createFeatures(text: String): FloatArray {
        val vector = FloatArray(featureSize)
        val normalized = normalize(text)
        val tokens = tokenize(normalized)
            .map(::stem)
            .filter { it.isNotBlank() && it !in STOP_WORDS }

        if (tokens.isEmpty()) {
            vector[featureSize - 1] = 1f
            return vector
        }

        // Word features.
        for (token in tokens) {
            val bucket = positiveHash("w:$token", WORD_BUCKETS)
            vector[bucket] += 1.0f
        }

        // Word bigrams capture phrases such as "how to", "firebase auth",
        // "jetpack compose", "machine learning", etc.
        for (i in 0 until tokens.size - 1) {
            val phrase = "${tokens[i]}_${tokens[i + 1]}"
            val bucket = WORD_BUCKETS + positiveHash("b:$phrase", BIGRAM_BUCKETS)
            vector[bucket] += 1.35f
        }

        // Character n-grams help with spelling variation:
        // "programming", "program", "programmng", etc. share fragments.
        val compact = normalized.filter { it.isLetterOrDigit() }
        for (i in 0 until compact.length - 2) {
            val trigram = compact.substring(i, i + 3)
            val bucket = WORD_BUCKETS + BIGRAM_BUCKETS +
                positiveHash("c:$trigram", CHAR_BUCKETS)
            vector[bucket] += 0.35f
        }

        // Extra linguistic features.
        val e = featureSize - EXTRA_FEATURES
        vector[e] = tokens.size.coerceAtMost(100) / 100f
        vector[e + 1] = normalized.length.coerceAtMost(500) / 500f
        vector[e + 2] = if (normalized.contains("?")) 1f else 0f
        vector[e + 3] = if (normalized.contains("!")) 1f else 0f
        vector[e + 4] = if (tokens.any { it in POSITIVE }) 1f else 0f
        vector[e + 5] = if (tokens.any { it in NEGATIVE }) 1f else 0f
        vector[e + 6] = if (tokens.any { it == "code" || it == "coding" }) 1f else 0f
        vector[e + 7] = if (tokens.any { it == "android" }) 1f else 0f
        vector[e + 8] = if (tokens.any { it == "kotlin" }) 1f else 0f
        vector[e + 9] = if (tokens.any { it == "firebase" }) 1f else 0f
        vector[e + 10] = if (tokens.any { it == "ai" || it == "artificial" }) 1f else 0f
        vector[e + 11] = if (tokens.any { it == "machine" || it == "learning" }) 1f else 0f
        vector[e + 12] = if (tokens.any { it == "math" || it == "calculate" }) 1f else 0f
        vector[e + 13] = if (tokens.any { it == "help" || it == "explain" }) 1f else 0f
        vector[e + 14] = if (tokens.any { it == "hello" || it == "hi" || it == "hey" }) 1f else 0f
        vector[e + 15] = if (tokens.any { it == "why" }) 1f else 0f
        vector[e + 16] = if (tokens.any { it == "how" }) 1f else 0f
        vector[e + 17] = if (tokens.any { it == "what" }) 1f else 0f
        vector[e + 18] = if (tokens.any { it == "where" }) 1f else 0f
        vector[e + 19] = if (tokens.any { it == "when" }) 1f else 0f
        vector[e + 20] = if (tokens.any { it == "error" || it == "exception" }) 1f else 0f
        vector[e + 21] = if (tokens.any { it == "database" || it == "firestore" || it == "sql" }) 1f else 0f
        vector[e + 22] = if (tokens.any { it == "compose" || it == "ui" }) 1f else 0f
        vector[e + 23] = if (tokens.any { it == "network" || it == "api" }) 1f else 0f
        vector[e + 24] = if (tokens.any { it == "text" || it == "sentence" }) 1f else 0f
        vector[e + 25] = if (tokens.any { it == "learn" || it == "teach" }) 1f else 0f
        vector[e + 26] = if (tokens.any { it == "app" || it == "application" }) 1f else 0f
        vector[e + 27] = if (tokens.any { it == "bug" || it == "debug" }) 1f else 0f
        vector[e + 28] = if (tokens.any { it == "thanks" || it == "thank" }) 1f else 0f
        vector[e + 29] = sentimentScore(tokens).toFloat()
        vector[e + 30] = averageTokenLength(tokens) / 12f
        vector[e + 31] = 1f // bias-like constant feature

        // L2 normalization improves stability for the neural network.
        var sumSquares = 0.0
        for (value in vector) sumSquares += value * value
        val norm = sqrt(sumSquares).toFloat().coerceAtLeast(1e-6f)
        for (i in vector.indices) vector[i] /= norm

        return vector
    }

    fun normalize(text: String): String =
        text
            .lowercase(Locale.getDefault())
            .replace(Regex("[^\\p{L}\\p{N}\\s?!.'_-]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

    fun tokenize(text: String): List<String> =
        Regex("[\\p{L}\\p{N}']+").findAll(text).map { it.value }.toList()

    /**
     * Small deterministic stemmer. It is intentionally conservative so that
     * words do not get damaged too aggressively.
     */
    fun stem(word: String): String {
        if (word.length <= 4) return word
        var w = word
        val suffixes = listOf("ingly", "edly", "ing", "ed", "ies", "es", "s")
        for (suffix in suffixes) {
            if (w.endsWith(suffix) && w.length - suffix.length >= 3) {
                w = w.removeSuffix(suffix)
                break
            }
        }
        return w
    }

    fun sentimentScore(text: String): Double {
        val tokens = tokenize(normalize(text)).map(::stem)
        return sentimentScore(tokens)
    }

    private fun sentimentScore(tokens: List<String>): Double {
        var score = 0
        for (token in tokens) {
            if (token in POSITIVE) score++
            if (token in NEGATIVE) score--
        }
        return score.coerceIn(-3, 3) / 3.0
    }

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size)
        var dot = 0.0
        var aa = 0.0
        var bb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            aa += a[i] * a[i]
            bb += b[i] * b[i]
        }
        if (aa == 0.0 || bb == 0.0) return 0.0
        return dot / (sqrt(aa) * sqrt(bb))
    }

    private fun averageTokenLength(tokens: List<String>): Float =
        if (tokens.isEmpty()) 0f
        else tokens.sumOf { it.length }.toFloat() / tokens.size

    /**
     * Fast deterministic FNV-style hash. The absolute value problem of
     * Int.MIN_VALUE is avoided by masking the sign bit.
     */
    private fun positiveHash(value: String, buckets: Int): Int {
        var h = 0x811C9DC5.toInt()
        for (c in value) {
            h = h xor c.code
            h *= 16777619
        }
        return (h and Int.MAX_VALUE) % buckets
    }
}
