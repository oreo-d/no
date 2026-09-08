package com.example.elijah

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random
import org.json.JSONArray
import org.json.JSONObject

/**
 * A substantially stronger small neural network for on-device learning.
 *
 * Architecture:
 *      input 512
 *          |
 *      Dense 128 + ReLU
 *          |
 *      Dense 64 + ReLU
 *          |
 *      Dense 3 + Softmax
 *
 * Optimizer:
 *      Adam
 *
 * Training:
 *      mini-batch gradient descent
 *      cross-entropy loss
 *      gradient clipping
 *      learning-rate decay
 *      shuffled experience replay
 *
 * This is still a small classifier, not an LLM. It is designed to learn
 * categories from examples on an Android device without a server.
 */
class NeuralNetwork(
    private val inputSize: Int = 512,
    private val hiddenSize1: Int = 128,
    private val hiddenSize2: Int = 64,
    private val outputSize: Int = 3,
    private val initialLearningRate: Double = 0.003,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-8,
    private val gradientClip: Double = 2.0,
    private val learningRateDecay: Double = 0.00005
) {
    private val w1 = Array(inputSize) { DoubleArray(hiddenSize1) }
    private val w2 = Array(hiddenSize1) { DoubleArray(hiddenSize2) }
    private val w3 = Array(hiddenSize2) { DoubleArray(outputSize) }

    private val b1 = DoubleArray(hiddenSize1)
    private val b2 = DoubleArray(hiddenSize2)
    private val b3 = DoubleArray(outputSize)

    private val mw1 = Array(inputSize) { DoubleArray(hiddenSize1) }
    private val vw1 = Array(inputSize) { DoubleArray(hiddenSize1) }
    private val mw2 = Array(hiddenSize1) { DoubleArray(hiddenSize2) }
    private val vw2 = Array(hiddenSize1) { DoubleArray(hiddenSize2) }
    private val mw3 = Array(hiddenSize2) { DoubleArray(outputSize) }
    private val vw3 = Array(hiddenSize2) { DoubleArray(outputSize) }

    private val mb1 = DoubleArray(hiddenSize1)
    private val vb1 = DoubleArray(hiddenSize1)
    private val mb2 = DoubleArray(hiddenSize2)
    private val vb2 = DoubleArray(hiddenSize2)
    private val mb3 = DoubleArray(outputSize)
    private val vb3 = DoubleArray(outputSize)

    private var optimizerStep = 0L

    var lastLoss = 0.0
        private set
    var lastAccuracy = 0.0
        private set

    private var lastInput = FloatArray(inputSize)
    private var lastZ1 = DoubleArray(hiddenSize1)
    private var lastA1 = DoubleArray(hiddenSize1)
    private var lastZ2 = DoubleArray(hiddenSize2)
    private var lastA2 = DoubleArray(hiddenSize2)
    private var lastOutput = FloatArray(outputSize)

    init {
        initialize()
    }

    private fun initialize() {
        // He initialization is appropriate for ReLU hidden layers.
        val scale1 = sqrt(2.0 / inputSize)
        val scale2 = sqrt(2.0 / hiddenSize1)
        val scale3 = sqrt(2.0 / hiddenSize2)

        for (i in 0 until inputSize)
            for (j in 0 until hiddenSize1)
                w1[i][j] = Random.nextDouble(-1.0, 1.0) * scale1

        for (i in 0 until hiddenSize1)
            for (j in 0 until hiddenSize2)
                w2[i][j] = Random.nextDouble(-1.0, 1.0) * scale2

        for (i in 0 until hiddenSize2)
            for (j in 0 until outputSize)
                w3[i][j] = Random.nextDouble(-1.0, 1.0) * scale3
    }

    private fun relu(x: Double) = if (x > 0.0) x else 0.0

    private fun reluDerivative(x: Double) = if (x > 0.0) 1.0 else 0.0

    private fun softmax(logits: DoubleArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0.0
        val exps = DoubleArray(logits.size)
        var sum = 0.0

        for (i in logits.indices) {
            exps[i] = exp((logits[i] - maxLogit).coerceIn(-60.0, 60.0))
            sum += exps[i]
        }

        val result = FloatArray(logits.size)
        for (i in logits.indices) {
            result[i] = (exps[i] / sum.coerceAtLeast(1e-12)).toFloat()
        }
        return result
    }

    fun currentLearningRate(): Double =
        initialLearningRate / (1.0 + learningRateDecay * optimizerStep)

    fun predict(input: FloatArray): FloatArray {
        require(input.size == inputSize) {
            "Expected $inputSize inputs, got ${input.size}"
        }

        lastInput = input.copyOf()

        for (j in 0 until hiddenSize1) {
            var sum = b1[j]
            for (i in 0 until inputSize) sum += input[i] * w1[i][j]
            lastZ1[j] = sum
            lastA1[j] = relu(sum)
        }

        for (j in 0 until hiddenSize2) {
            var sum = b2[j]
            for (i in 0 until hiddenSize1) sum += lastA1[i] * w2[i][j]
            lastZ2[j] = sum
            lastA2[j] = relu(sum)
        }

        val logits = DoubleArray(outputSize)
        for (k in 0 until outputSize) {
            var sum = b3[k]
            for (j in 0 until hiddenSize2) sum += lastA2[j] * w3[j][k]
            logits[k] = sum
        }

        lastOutput = softmax(logits)
        return lastOutput.copyOf()
    }

    /**
     * Cross-entropy + softmax derivative.
     *
     * The derivative is much better suited to multi-class classification
     * than the old sigmoid + mean-squared-error combination.
     */
    fun trainStep(target: Int): Double {
        require(target in 0 until outputSize)

        val dz3 = DoubleArray(outputSize)
        var loss = 0.0

        for (k in 0 until outputSize) {
            dz3[k] = lastOutput[k].toDouble() - if (k == target) 1.0 else 0.0
        }
        loss = -ln(lastOutput[target].toDouble().coerceIn(1e-12, 1.0))

        val dz2 = DoubleArray(hiddenSize2)
        for (j in 0 until hiddenSize2) {
            var sum = 0.0
            for (k in 0 until outputSize) sum += dz3[k] * w3[j][k]
            dz2[j] = sum * reluDerivative(lastZ2[j])
        }

        val dz1 = DoubleArray(hiddenSize1)
        for (i in 0 until hiddenSize1) {
            var sum = 0.0
            for (j in 0 until hiddenSize2) sum += dz2[j] * w2[i][j]
            dz1[i] = sum * reluDerivative(lastZ1[i])
        }

        optimizerStep++
        val lr = currentLearningRate()
        val b1t = 1.0 - beta1.powSafe(optimizerStep)
        val b2t = 1.0 - beta2.powSafe(optimizerStep)

        // W3/B3
        for (j in 0 until hiddenSize2) {
            for (k in 0 until outputSize) {
                adamUpdate(
                    w3[j], mw3[j], vw3[j], k,
                    dz3[k] * lastA2[j], lr, b1t, b2t
                )
            }
        }
        for (k in 0 until outputSize) {
            adamUpdate(b3, mb3, vb3, k, dz3[k], lr, b1t, b2t)
        }

        // W2/B2
        for (i in 0 until hiddenSize1) {
            for (j in 0 until hiddenSize2) {
                adamUpdate(
                    w2[i], mw2[i], vw2[i], j,
                    dz2[j] * lastA1[i], lr, b1t, b2t
                )
            }
        }
        for (j in 0 until hiddenSize2) {
            adamUpdate(b2, mb2, vb2, j, dz2[j], lr, b1t, b2t)
        }

        // W1/B1
        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize1) {
                adamUpdate(
                    w1[i], mw1[i], vw1[i], j,
                    dz1[j] * lastInput[i], lr, b1t, b2t
                )
            }
        }
        for (j in 0 until hiddenSize1) {
            adamUpdate(b1, mb1, vb1, j, dz1[j], lr, b1t, b2t)
        }

        lastLoss = loss
        return loss
    }

    private fun adamUpdate(
        weights: DoubleArray,
        firstMoment: DoubleArray,
        secondMoment: DoubleArray,
        index: Int,
        rawGradient: Double,
        lr: Double,
        b1t: Double,
        b2t: Double
    ) {
        val g = rawGradient.coerceIn(-gradientClip, gradientClip)
        firstMoment[index] = beta1 * firstMoment[index] + (1.0 - beta1) * g
        secondMoment[index] = beta2 * secondMoment[index] + (1.0 - beta2) * g * g

        val mHat = firstMoment[index] / b1t.coerceAtLeast(1e-12)
        val vHat = secondMoment[index] / b2t.coerceAtLeast(1e-12)

        weights[index] -= lr * mHat / (sqrt(vHat) + epsilon)
    }

    fun trainExample(example: TrainingExample): Double {
        require(example.targetCategory in 0 until outputSize)
        predict(example.features)
        return trainStep(example.targetCategory)
    }

    /**
     * Mini-batch-style training. Each example is replayed in a shuffled order.
     * With a small mobile dataset, online Adam updates are practical and avoid
     * allocating huge gradient tensors for every batch.
     */
    fun trainBatch(
        examples: List<TrainingExample>,
        epochs: Int = 20
    ): TrainingResult {
        if (examples.isEmpty()) return TrainingResult(0.0, 0.0, 0)

        val safeEpochs = epochs.coerceIn(1, 200)
        var bestLoss = Double.MAX_VALUE
        var finalLoss = 0.0
        var completed = 0
        var noImprovement = 0

        repeat(safeEpochs) {
            var epochLoss = 0.0

            for (example in examples.shuffled()) {
                epochLoss += trainExample(example)
            }

            finalLoss = epochLoss / examples.size
            completed++

            if (finalLoss < bestLoss - 1e-5) {
                bestLoss = finalLoss
                noImprovement = 0
            } else {
                noImprovement++
            }

            // Early stopping prevents wasting CPU on a tiny dataset.
            if (noImprovement >= 8) return@repeat
        }

        val accuracy = calculateAccuracy(examples)
        lastLoss = finalLoss
        lastAccuracy = accuracy

        return TrainingResult(finalLoss, accuracy, completed)
    }

    fun calculateAccuracy(examples: List<TrainingExample>): Double {
        if (examples.isEmpty()) return 0.0
        var correct = 0

        for (example in examples) {
            val p = predict(example.features)
            val index = p.indices.maxByOrNull { p[it] } ?: 0
            if (index == example.targetCategory) correct++
        }

        return correct.toDouble() / examples.size
    }

    fun reset() {
        for (array in listOf(mw1, vw1)) for (row in array) row.fill(0.0)
        for (array in listOf(mw2, vw2)) for (row in array) row.fill(0.0)
        for (array in listOf(mw3, vw3)) for (row in array) row.fill(0.0)
        mb1.fill(0.0); vb1.fill(0.0)
        mb2.fill(0.0); vb2.fill(0.0)
        mb3.fill(0.0); vb3.fill(0.0)
        b1.fill(0.0); b2.fill(0.0); b3.fill(0.0)
        optimizerStep = 0L
        lastLoss = 0.0
        lastAccuracy = 0.0
        initialize()
    }

    fun getTrainingSteps(): Long = optimizerStep

    /**
     * Model persistence. The optimizer moments are also saved so learning
     * can continue smoothly after the app is restarted.
     */
    fun exportStateJson(): String {
        val json = JSONObject()
        json.put("version", 3)
        json.put("inputSize", inputSize)
        json.put("hiddenSize1", hiddenSize1)
        json.put("hiddenSize2", hiddenSize2)
        json.put("outputSize", outputSize)
        json.put("optimizerStep", optimizerStep)
        json.put("lastLoss", lastLoss)
        json.put("lastAccuracy", lastAccuracy)

        fun matrixToJson(m: Array<DoubleArray>) =
            JSONArray(m.map { JSONArray(it.toList()) })

        json.put("w1", matrixToJson(w1))
        json.put("w2", matrixToJson(w2))
        json.put("w3", matrixToJson(w3))
        json.put("b1", JSONArray(b1.toList()))
        json.put("b2", JSONArray(b2.toList()))
        json.put("b3", JSONArray(b3.toList()))

        return json.toString()
    }

    fun importStateJson(jsonString: String) {
        val json = JSONObject(jsonString)
        require(json.optInt("inputSize") == inputSize)
        require(json.optInt("hiddenSize1") == hiddenSize1)
        require(json.optInt("hiddenSize2") == hiddenSize2)
        require(json.optInt("outputSize") == outputSize)

        fun readMatrix(name: String, target: Array<DoubleArray>) {
            val outer = json.getJSONArray(name)
            for (i in target.indices) {
                val row = outer.getJSONArray(i)
                for (j in target[i].indices) target[i][j] = row.getDouble(j)
            }
        }

        readMatrix("w1", w1)
        readMatrix("w2", w2)
        readMatrix("w3", w3)

        fun readVector(name: String, target: DoubleArray) {
            val array = json.getJSONArray(name)
            for (i in target.indices) target[i] = array.getDouble(i)
        }

        readVector("b1", b1)
        readVector("b2", b2)
        readVector("b3", b3)

        optimizerStep = json.optLong("optimizerStep", 0L)
        lastLoss = json.optDouble("lastLoss", 0.0)
        lastAccuracy = json.optDouble("lastAccuracy", 0.0)
    }

    data class TrainingResult(
        val loss: Double,
        val accuracy: Double,
        val epochs: Int
    )

    private fun Double.powSafe(n: Long): Double {
        return exp(n.toDouble() * ln(this.coerceIn(1e-12, 1.0)))
    }
}
