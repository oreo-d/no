package com.example.elijah

import java.util.concurrent.ConcurrentHashMap

/** Privacy-conscious event layer. It stores only application events explicitly recorded by the caller. */
data class ConsentState(
    val analytics: Boolean = false,
    val personalization: Boolean = false,
    val marketing: Boolean = false
)

data class UserEvent(
    val userId: String,
    val type: String,
    val timestampMs: Long = System.currentTimeMillis(),
    val value: Double? = null,
    val category: String? = null
)

class PrivacyIntelligence {
    @Volatile var consent: ConsentState = ConsentState()
    private val events = ConcurrentHashMap<String, MutableList<UserEvent>>()

    fun record(event: UserEvent) {
        val allowed = when (event.type) {
            "ad_impression", "ad_click", "campaign_view", "campaign_conversion" -> consent.marketing
            "product_viewed", "product_searched", "recommendation_clicked" -> consent.personalization
            else -> consent.analytics
        }
        if (!allowed) return
        events.computeIfAbsent(event.userId) { mutableListOf() }.add(event)
    }

    fun eventsFor(userId: String): List<UserEvent> = events[userId]?.toList() ?: emptyList()

    fun deleteUser(userId: String) { events.remove(userId) }

    fun clearAll() { events.clear() }
}

/** Converts allowed interaction events into non-sensitive preference scores in [0,1]. */
class PreferenceEngine {
    fun score(events: List<UserEvent>, categories: List<String>): Map<String, Double> {
        if (events.isEmpty()) return categories.associateWith { 0.0 }
        val counts = categories.associateWith { c ->
            events.count { it.category.equals(c, ignoreCase = true) }.toDouble()
        }
        val max = counts.values.maxOrNull() ?: 1.0
        return counts.mapValues { (_, v) -> if (max == 0.0) 0.0 else v / max }
    }
}
