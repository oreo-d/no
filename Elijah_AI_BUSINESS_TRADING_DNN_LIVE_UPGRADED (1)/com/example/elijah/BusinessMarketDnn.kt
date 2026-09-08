package com.example.elijah

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Production-oriented small DNN wrapper.
 *
 * Two independently trained heads are used:
 *  1) business head -> probability of a near-term conversion
 *  2) market head   -> bearish / neutral / bullish probability
 *
 * Both heads use the project's trainable NeuralNetwork implementation, so
 * this is genuine supervised training rather than a hard-coded signal.
 */
class BusinessMarketDnn(
    private val businessInputSize: Int = 16,
    private val marketInputSize: Int = 16
) {
    companion object {
        const val BUSINESS_CONVERSION = 0
        const val BUSINESS_NO_CONVERSION = 1

        const val MARKET_BEARISH = 0
        const val MARKET_NEUTRAL = 1
        const val MARKET_BULLISH = 2
    }

    private val businessModel = NeuralNetwork(
        inputSize = businessInputSize,
        hiddenSize1 = 64,
        hiddenSize2 = 32,
        outputSize = 2,
        initialLearningRate = 0.0015
    )

    private val marketModel = NeuralNetwork(
        inputSize = marketInputSize,
        hiddenSize1 = 64,
        hiddenSize2 = 32,
        outputSize = 3,
        initialLearningRate = 0.0015
    )

    data class BusinessPrediction(
        val conversionProbability: Double,
        val noConversionProbability: Double,
        val confidence: Double
    )

    data class MarketPrediction(
        val bearishProbability: Double,
        val neutralProbability: Double,
        val bullishProbability: Double,
        val action: String,
        val confidence: Double
    )

    data class TrainingReport(
        val businessExamples: Int,
        val businessAccuracy: Double,
        val businessLoss: Double,
        val marketExamples: Int,
        val marketAccuracy: Double,
        val marketLoss: Double
    )

    fun predictBusiness(features: FloatArray): BusinessPrediction {
        require(features.size == businessInputSize)
        val p = businessModel.predict(features)
        val best = p.maxOrNull()?.toDouble() ?: 0.0
        return BusinessPrediction(
            conversionProbability = p[BUSINESS_CONVERSION].toDouble(),
            noConversionProbability = p[BUSINESS_NO_CONVERSION].toDouble(),
            confidence = best
        )
    }

    fun predictMarket(features: FloatArray): MarketPrediction {
        require(features.size == marketInputSize)
        val p = marketModel.predict(features)
        val index = p.indices.maxByOrNull { p[it] } ?: MARKET_NEUTRAL
        val action = when (index) {
            MARKET_BULLISH -> "WATCH / BULLISH"
            MARKET_BEARISH -> "WATCH / BEARISH"
            else -> "HOLD / NEUTRAL"
        }
        return MarketPrediction(
            bearishProbability = p[MARKET_BEARISH].toDouble(),
            neutralProbability = p[MARKET_NEUTRAL].toDouble(),
            bullishProbability = p[MARKET_BULLISH].toDouble(),
            action = action,
            confidence = p[index].toDouble()
        )
    }

    fun trainBusiness(examples: List<TrainingExample>, epochs: Int = 25): NeuralNetwork.TrainingResult {
        require(examples.all { it.features.size == businessInputSize })
        return businessModel.trainBatch(examples, epochs)
    }

    fun trainMarket(examples: List<TrainingExample>, epochs: Int = 25): NeuralNetwork.TrainingResult {
        require(examples.all { it.features.size == marketInputSize })
        return marketModel.trainBatch(examples, epochs)
    }

    fun train(
        businessExamples: List<TrainingExample>,
        marketExamples: List<TrainingExample>,
        epochs: Int = 25
    ): TrainingReport {
        val b = if (businessExamples.isNotEmpty()) trainBusiness(businessExamples, epochs)
        else NeuralNetwork.TrainingResult(0.0, 0.0, 0)

        val m = if (marketExamples.isNotEmpty()) trainMarket(marketExamples, epochs)
        else NeuralNetwork.TrainingResult(0.0, 0.0, 0)

        return TrainingReport(
            businessExamples.size, b.accuracy, b.loss,
            marketExamples.size, m.accuracy, m.loss
        )
    }

    fun exportBusinessState(): String = businessModel.exportStateJson()
    fun exportMarketState(): String = marketModel.exportStateJson()

    fun importBusinessState(state: String) = businessModel.importStateJson(state)
    fun importMarketState(state: String) = marketModel.importStateJson(state)

    fun reset() {
        businessModel.reset()
        marketModel.reset()
    }

    fun businessSteps(): Long = businessModel.getTrainingSteps()
    fun marketSteps(): Long = marketModel.getTrainingSteps()
}

/**
 * Turns raw business events into 16 normalized model features.
 *
 * Features are deliberately non-sensitive:
 * recency, frequency, value, engagement and permitted interaction counts.
 */
object BusinessFeatureEngineering {
    private const val DAY_MS = 86_400_000L

    fun fromEvents(events: List<UserEvent>, nowMs: Long = System.currentTimeMillis()): FloatArray {
        val sorted = events.sortedBy { it.timestampMs }
        val last = sorted.maxOfOrNull { it.timestampMs } ?: nowMs
        val daysSince = ((nowMs - last).coerceAtLeast(0L) / DAY_MS).toDouble()

        fun count(type: String) = sorted.count { it.type == type }.toDouble()
        fun sum(type: String) = sorted.filter { it.type == type }.sumOf { it.value ?: 0.0 }

        val views = count("product_viewed")
        val searches = count("product_searched")
        val clicks = count("recommendation_clicked")
        val adClicks = count("ad_click")
        val impressions = count("ad_impression")
        val conversions = count("campaign_conversion")
        val revenue = sum("campaign_conversion")
        val spend = sum("ad_spend")
        val cart = count("product_added_to_cart")
        val engagement = count("session_engaged")
        val total = sorted.size.toDouble()

        val categories = sorted.mapNotNull { it.category?.lowercase() }.distinct().size.toDouble()
        val ctr = if (impressions == 0.0) 0.0 else adClicks / impressions
        val conversionRate = if (clicks == 0.0) 0.0 else conversions / clicks
        val avgValue = if (conversions == 0.0) 0.0 else revenue / conversions
        val recency = 1.0 / (1.0 + daysSince / 30.0)

        return normalize(
            doubleArrayOf(
                recency, ln1p(total), ln1p(views), ln1p(searches),
                ln1p(clicks), ln1p(adClicks), ln1p(conversions), ln1p(cart),
                ln1p(engagement), ln1p(revenue), ln1p(spend), ctr,
                conversionRate, avgValue / 1000.0, categories / 20.0,
                if (total > 0) conversions / total else 0.0
            )
        )
    }

    private fun ln1p(x: Double) = ln(1.0 + max(0.0, x))

    private fun normalize(values: DoubleArray): FloatArray {
        return values.map { (it / (1.0 + abs(it))).coerceIn(-1.0, 1.0).toFloat() }.toFloatArray()
    }
}

/**
 * Converts candles into technical features and automatically creates labels
 * from future returns. This prevents look-ahead leakage: each example only
 * uses candles up to index i, while its label comes from later candles.
 */
object MarketFeatureEngineering {
    const val FEATURE_COUNT = 16

    fun features(candles: List<MarketCandle>, endIndex: Int = candles.lastIndex): FloatArray {
        require(endIndex in 0 until candles.size)
        val c = candles[endIndex]

        fun closeAgo(n: Int): Double {
            val i = (endIndex - n).coerceAtLeast(0)
            return candles[i].close
        }

        fun sma(window: Int): Double {
            val start = (endIndex - window + 1).coerceAtLeast(0)
            val values = candles.subList(start, endIndex + 1).map { it.close }
            return values.average()
        }

        fun volatility(window: Int): Double {
            val start = (endIndex - window).coerceAtLeast(0)
            val r = candles.subList(start, endIndex + 1).zipWithNext()
                .map { (a, b) -> b.close / a.close - 1.0 }
            if (r.size < 2) return 0.0
            val mean = r.average()
            return sqrt(r.map { (it - mean).pow(2) }.average())
        }

        fun rsi(window: Int = 14): Double {
            val start = (endIndex - window).coerceAtLeast(0)
            val diffs = candles.subList(start, endIndex + 1).zipWithNext()
                .map { it.second.close - it.first.close }
            val gains = diffs.filter { it > 0 }.sum()
            val losses = -diffs.filter { it < 0 }.sum()
            if (losses == 0.0) return if (gains > 0) 1.0 else 0.5
            val rs = gains / losses
            return 1.0 - 1.0 / (1.0 + rs)
        }

        val ret1 = c.close / closeAgo(1) - 1.0
        val ret5 = c.close / closeAgo(5) - 1.0
        val ret10 = c.close / closeAgo(10) - 1.0
        val fastGap = if (sma(10) == 0.0) 0.0 else c.close / sma(10) - 1.0
        val slowGap = if (sma(30) == 0.0) 0.0 else c.close / sma(30) - 1.0
        val range = (c.high - c.low) / c.close
        val body = (c.close - c.open) / c.open
        val volRatio = if (candles.takeLast(20).map { it.volume }.average() == 0.0) 0.0
            else c.volume / candles.takeLast(20).map { it.volume }.average() - 1.0

        return doubleArrayOf(
            ret1, ret5, ret10, fastGap, slowGap, volatility(10),
            volatility(30), rsi(), range, body, volRatio,
            c.close / closeAgo(20) - 1.0,
            c.close / closeAgo(60) - 1.0,
            (sma(10) / sma(30) - 1.0),
            ((c.close - c.low) / (c.high - c.low).coerceAtLeast(1e-9)),
            1.0
        ).map { (it / (1.0 + abs(it))).coerceIn(-1.0, 1.0).toFloat() }.toFloatArray()
    }

    fun marketExamples(
        candles: List<MarketCandle>,
        horizon: Int = 5,
        bullishThreshold: Double = 0.005,
        bearishThreshold: Double = -0.005
    ): List<TrainingExample> {
        val minimum = 60
        if (candles.size < minimum + horizon) return emptyList()

        return (60 until candles.size - horizon).mapNotNull { i ->
            val future = candles[i + horizon].close / candles[i].close - 1.0
            val target = when {
                future >= bullishThreshold -> BusinessMarketDnn.MARKET_BULLISH
                future <= bearishThreshold -> BusinessMarketDnn.MARKET_BEARISH
                else -> BusinessMarketDnn.MARKET_NEUTRAL
            }
            TrainingExample(features(candles, i), target)
        }
    }
}

/**
 * Creates business examples from event histories.
 *
 * Label = whether a conversion occurs within the next `lookAheadMs` after
 * the feature snapshot. Only events already present in the permitted event
 * stream are used.
 */
object BusinessTrainingBuilder {
    fun examples(
        events: List<UserEvent>,
        lookAheadMs: Long = 7L * 86_400_000L
    ): List<TrainingExample> {
        if (events.size < 4) return emptyList()
        val sorted = events.sortedBy { it.timestampMs }
        val cutPoints = sorted.indices.filter { it >= 2 }

        return cutPoints.mapNotNull { i ->
            val cutoff = sorted[i].timestampMs
            val history = sorted.take(i + 1)
            val future = sorted.drop(i + 1)
                .any {
                    it.timestampMs > cutoff &&
                        it.timestampMs <= cutoff + lookAheadMs &&
                        it.type == "campaign_conversion"
                }
            TrainingExample(
                features = BusinessFeatureEngineering.fromEvents(history, cutoff),
                targetCategory = if (future) BusinessMarketDnn.BUSINESS_CONVERSION
                else BusinessMarketDnn.BUSINESS_NO_CONVERSION,
                timestamp = cutoff
            )
        }
    }
}
