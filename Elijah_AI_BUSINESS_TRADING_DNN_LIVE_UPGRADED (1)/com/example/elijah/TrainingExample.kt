package com.example.elijah

data class TrainingExample(
    val features: FloatArray,
    val targetCategory: Int,
    val timestamp: Long = System.currentTimeMillis()
) {
    init {
        require(features.isNotEmpty())
        require(targetCategory >= 0)
    }

    fun copyExample(): TrainingExample =
        TrainingExample(features.copyOf(), targetCategory, timestamp)
}
