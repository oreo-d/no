package com.example.elijah

/** Central registry exposing Elijah AI's classical ML and advanced AI algorithms. */
object AlgorithmSuite {
    val linearRegression = LinearRegression()
    val decisionTree = DecisionTree()
    val svm = SupportVectorMachine()
    val kMeans = KMeans(2)
    val pca = PCA(2)
    val qLearning = QLearning(states=16, actions=4)
    fun transformerAttention(dimension:Int,tokens:Array<FloatArray>) =
        TransformerEncoder(dimension).selfAttention(tokens)
}
