package com.example.elijah

import kotlin.math.sqrt

/**
 * Lightweight deterministic sentence embedding.
 *
 * This is not a pretrained transformer embedding. It creates dense,
 * normalized semantic-ish vectors from word/character hashing plus
 * positional features, making it fully offline and dependency-free.
 */
class EmbeddingModel(private val dimension: Int = 384) {

    fun embed(text: String): FloatArray {
        require(dimension > 0)
        val vector = FloatArray(dimension)
        val tokens = tokenize(text)

        if (tokens.isEmpty()) return vector

        // Word hashing + subword hashing gives robustness to spelling variants.
        tokens.forEachIndexed { index, token ->
            addHash(vector, token, 1.0f)
            addHash(vector, "w:$token", 0.35f)

            val chars = "^$token$"
            if (chars.length >= 3) {
                for (i in 0..chars.length - 3) {
                    addHash(vector, "c:${chars.substring(i, i + 3)}", 0.22f)
                }
            }

            // A small positional signal distinguishes otherwise identical orderings.
            val posWeight = 1f / (1f + index * 0.12f)
            addHash(vector, "p:${index.coerceAtMost(15)}:$token", 0.08f * posWeight)
        }

        // Phrase hashing.
        for (i in 0 until tokens.lastIndex) {
            addHash(vector, "bg:${tokens[i]}_${tokens[i + 1]}", 0.65f)
        }

        normalize(vector)
        return vector
    }

    fun similarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Embedding dimensions must match." }
        var dot = 0f
        var aa = 0f
        var bb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            aa += a[i] * a[i]
            bb += b[i] * b[i]
        }
        if (aa <= 0f || bb <= 0f) return 0f
        return (dot / (sqrt(aa) * sqrt(bb))).coerceIn(-1f, 1f)
    }

    private fun addHash(vector: FloatArray, value: String, weight: Float) {
        val h = fnv1a(value)
        val index = (h and Int.MAX_VALUE) % vector.size
        val sign = if ((h ushr 1 and 1) == 0) 1f else -1f
        vector[index] += sign * weight
    }

    private fun normalize(vector: FloatArray) {
        var sum = 0f
        for (v in vector) sum += v * v
        val norm = sqrt(sum)
        if (norm > 1e-8f) {
            for (i in vector.indices) vector[i] /= norm
        }
    }

    private fun tokenize(text: String): List<String> =
        text.lowercase()
            .replace(Regex("[^\\p{L}\\p{N}_+#.-]+"), " ")
            .trim()
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }
            .map { it.trim('.', ',', '!', '?', ':', ';') }

    private fun fnv1a(value: String): Int {
        var h = 0x811C9DC5.toInt()
        for (c in value) {
            h = h xor c.code
            h *= 16777619
        }
        return h
    }
}
