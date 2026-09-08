package com.example.elijah

data class CustomerSegment(
    val name: String,
    val customerIds: List<String>,
    val averageValue: Double,
    val engagementRate: Double
)

data class CampaignMetrics(
    val impressions: Long,
    val clicks: Long,
    val conversions: Long,
    val revenue: Double,
    val spend: Double
) {
    val ctr: Double get() = if (impressions == 0L) 0.0 else clicks.toDouble() / impressions
    val conversionRate: Double get() = if (clicks == 0L) 0.0 else conversions.toDouble() / clicks
    val roas: Double get() = if (spend == 0.0) 0.0 else revenue / spend
}

class BusinessIntelligence {
    fun campaign(impressions: Long, clicks: Long, conversions: Long, revenue: Double, spend: Double) =
        CampaignMetrics(impressions, clicks, conversions, revenue, spend)

    /** Simple RFM-style segmentation using normalized recency, frequency and monetary scores. */
    fun segment(
        customers: List<CustomerRecord>,
        recencyDaysCutoff: Int = 30
    ): List<CustomerSegment> {
        val active = customers.filter { it.recencyDays <= recencyDaysCutoff }
        val groups = active.groupBy {
            when {
                it.monetaryValue >= 1000 && it.frequency >= 10 -> "VIP"
                it.monetaryValue >= 500 || it.frequency >= 5 -> "Loyal"
                it.frequency <= 1 -> "New / Low activity"
                else -> "Growing"
            }
        }
        return groups.map { (name, list) ->
            CustomerSegment(
                name,
                list.map { it.id },
                list.map { it.monetaryValue }.average(),
                list.count { it.engaged }.toDouble() / list.size
            )
        }
    }
}

data class CustomerRecord(
    val id: String,
    val recencyDays: Int,
    val frequency: Int,
    val monetaryValue: Double,
    val engaged: Boolean
)

class RecommendationEngine {
    fun rank(
        candidates: List<RecommendationCandidate>,
        preferences: Map<String, Double>
    ): List<RecommendationCandidate> =
        candidates.sortedByDescending { candidate ->
            0.7 * (preferences[candidate.category] ?: 0.0) +
                0.3 * candidate.qualityScore.coerceIn(0.0, 1.0)
        }
}

data class RecommendationCandidate(
    val id: String,
    val category: String,
    val qualityScore: Double
)
