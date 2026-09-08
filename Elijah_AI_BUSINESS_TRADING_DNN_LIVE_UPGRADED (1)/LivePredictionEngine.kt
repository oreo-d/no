package com.example.elijah

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Live inference/training coordinator.
 *
 * It can:
 *  - train from historical business events and market candles
 *  - predict from the latest permitted business events
 *  - predict from the latest market candle
 *  - periodically refresh a market-data source and emit a new prediction
 *
 * It deliberately does NOT execute trades or send orders.
 */
class LivePredictionEngine(
    private val model: BusinessMarketDnn = BusinessMarketDnn()
) {
    data class LiveSnapshot(
        val timestampMs: Long,
        val business: BusinessMarketDnn.BusinessPrediction?,
        val market: BusinessMarketDnn.MarketPrediction?,
        val latestPrice: Double?,
        val marketDataPoints: Int
    )

    fun model(): BusinessMarketDnn = model

    fun train(
        events: List<UserEvent>,
        candles: List<MarketCandle>,
        epochs: Int = 25
    ): BusinessMarketDnn.TrainingReport {
        val businessExamples = BusinessTrainingBuilder.examples(events)
        val marketExamples = MarketFeatureEngineering.marketExamples(candles)
        return model.train(businessExamples, marketExamples, epochs)
    }

    fun predictBusiness(events: List<UserEvent>): BusinessMarketDnn.BusinessPrediction =
        model.predictBusiness(BusinessFeatureEngineering.fromEvents(events))

    fun predictMarket(candles: List<MarketCandle>): BusinessMarketDnn.MarketPrediction {
        require(candles.size >= 61) { "At least 61 candles are required for live market inference." }
        return model.predictMarket(MarketFeatureEngineering.features(candles))
    }

    fun snapshot(
        events: List<UserEvent>,
        candles: List<MarketCandle>
    ): LiveSnapshot {
        val market = if (candles.size >= 61) predictMarket(candles) else null
        return LiveSnapshot(
            timestampMs = System.currentTimeMillis(),
            business = if (events.isNotEmpty()) predictBusiness(events) else null,
            market = market,
            latestPrice = candles.lastOrNull()?.close,
            marketDataPoints = candles.size
        )
    }

    /**
     * Polls your backend/data provider. Use a sensible interval for the
     * selected timeframe and respect the provider's rate limits.
     */
    fun startMarketPolling(
        scope: CoroutineScope,
        source: MarketDataSource,
        symbol: String,
        intervalMs: Long = 60_000L,
        onPrediction: (LiveSnapshot) -> Unit,
        businessEvents: () -> List<UserEvent> = { emptyList() }
    ): Job {
        return scope.launch(Dispatchers.IO) {
            while (isActive) {
                runCatching {
                    val candles = source.latestCandles(symbol, 250)
                    if (candles.isNotEmpty()) {
                        onPrediction(snapshot(businessEvents(), candles))
                    }
                }
                delay(intervalMs.coerceAtLeast(5_000L))
            }
        }
    }
}
