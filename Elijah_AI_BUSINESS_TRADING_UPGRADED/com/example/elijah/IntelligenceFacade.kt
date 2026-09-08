package com.example.elijah

/**
 * Small integration example for an Android ViewModel/service.
 */
class IntelligenceFacade {
    val privacy = PrivacyIntelligence()
    val preferences = PreferenceEngine()
    val business = BusinessIntelligence()
    val recommendations = RecommendationEngine()
    val trading = TradingAnalytics()

    fun recordProductView(userId: String, category: String) {
        privacy.record(UserEvent(userId, "product_viewed", category = category))
    }

    fun getPreferences(userId: String): Map<String, Double> =
        preferences.score(privacy.eventsFor(userId), listOf("electronics", "fashion", "software", "printing"))

    fun rankProducts(
        userId: String,
        products: List<RecommendationCandidate>
    ): List<RecommendationCandidate> =
        recommendations.rank(products, getPreferences(userId))
}
