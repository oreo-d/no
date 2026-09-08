package com.example.elijah

/**
 * OHLCV candle supplied by a trusted market-data provider.
 *
 * Prices must be adjusted/normalized consistently by the provider.
 * This module is for prediction and paper analysis; it never places orders.
 */
data class MarketCandle(
    val timestampMs: Long,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val volume: Double = 0.0
) {
    init {
        require(open > 0.0 && high > 0.0 && low > 0.0 && close > 0.0)
        require(high >= maxOf(open, close))
        require(low <= minOf(open, close))
        require(volume >= 0.0)
    }
}

/**
 * Implement this with Firebase/Cloud Functions, your broker/exchange API,
 * or another vetted market-data service.
 */
interface MarketDataSource {
    suspend fun latestCandles(symbol: String, limit: Int = 250): List<MarketCandle>
}

/**
 * Minimal HTTP adapter for a JSON endpoint owned by your application/backend.
 *
 * Expected response:
 * {
 *   "candles": [
 *     {"timestampMs": 123, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100}
 *   ]
 * }
 *
 * Keep API credentials on your server, not inside the Android APK.
 */
class JsonMarketDataSource(
    private val endpointFor: (String, Int) -> String,
    private val headers: Map<String, String> = emptyMap()
) : MarketDataSource {
    override suspend fun latestCandles(symbol: String, limit: Int): List<MarketCandle> {
        val url = java.net.URL(endpointFor(symbol, limit))
        val connection = (url.openConnection() as java.net.HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 10_000
            readTimeout = 15_000
            headers.forEach { (key, value) -> setRequestProperty(key, value) }
        }

        try {
            if (connection.responseCode !in 200..299) {
                throw IllegalStateException("Market data HTTP ${connection.responseCode}")
            }
            val body = connection.inputStream.bufferedReader().use { it.readText() }
            val root = org.json.JSONObject(body)
            val array = root.getJSONArray("candles")
            return buildList {
                for (i in 0 until array.length()) {
                    val c = array.getJSONObject(i)
                    add(
                        MarketCandle(
                            timestampMs = c.getLong("timestampMs"),
                            open = c.getDouble("open"),
                            high = c.getDouble("high"),
                            low = c.getDouble("low"),
                            close = c.getDouble("close"),
                            volume = c.optDouble("volume", 0.0)
                        )
                    )
                }
            }.sortedBy { it.timestampMs }.takeLast(limit.coerceAtLeast(1))
        } finally {
            connection.disconnect()
        }
    }
}
