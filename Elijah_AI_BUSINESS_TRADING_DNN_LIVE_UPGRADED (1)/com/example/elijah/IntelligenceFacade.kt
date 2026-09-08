package com.example.elijah

/**
 * Integration facade for business personalization, analytics and live DNN
 * predictions. The DNN uses supervised historical labels and then performs
 * inference on newly arriving data.
 */
class IntelligenceFacade {
    val privacy = PrivacyIntelligence()
    val preferences = PreferenceEngine()
    val business = BusinessIntelligence()
    val recommendations = RecommendationEngine()
    val trading = TradingAnalytics()

    private val liveEngine = LivePredictionEngine()
    val dnn: BusinessMarketDnn get() = liveEngine.model()

    fun recordProductView(userId: String, category: String) {
        privacy.record(UserEvent(userId, "product_viewed", category = category))
    }

    fun getPreferences(userId: String): Map<String, Double> =
        preferences.score(
            privacy.eventsFor(userId),
            listOf("electronics", "fashion", "software", "printing")
        )

    fun rankProducts(
        userId: String,
        products: List<RecommendationCandidate>
    ): List<RecommendationCandidate> =
        recommendations.rank(products, getPreferences(userId))

    /** Train both DNN heads on permitted historical business events + candles. */
    fun trainDnn(
        userId: String,
        candles: List<MarketCandle>,
        epochs: Int = 25
    ): BusinessMarketDnn.TrainingReport =
        liveEngine.train(privacy.eventsFor(userId), candles, epochs)

    /** Run inference using the newest business events and market candles. */
    fun predictLive(
        userId: String,
        candles: List<MarketCandle>
    ): LivePredictionEngine.LiveSnapshot =
        liveEngine.snapshot(privacy.eventsFor(userId), candles)

    fun model(): BusinessMarketDnn = liveEngine.model()
}
