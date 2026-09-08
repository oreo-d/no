package com.example.elijah

/**
 * Optional Firebase adapter contract.
 *
 * Keep Firebase SDK code behind this interface so the core AI package remains
 * dependency-free. Implement it with Firebase Analytics/Firestore in the app module.
 */
interface BusinessEventSink {
    suspend fun writeEvent(event: UserEvent)
    suspend fun deleteUser(userId: String)
}

class LocalBusinessEventSink(private val intelligence: PrivacyIntelligence) : BusinessEventSink {
    override suspend fun writeEvent(event: UserEvent) = intelligence.record(event)
    override suspend fun deleteUser(userId: String) = intelligence.deleteUser(userId)
}

/*
Example app-module implementation (requires Firebase dependencies):

class FirestoreBusinessEventSink(
    private val db: FirebaseFirestore
) : BusinessEventSink {
    override suspend fun writeEvent(event: UserEvent) {
        db.collection("events").add(
            mapOf(
                "userId" to event.userId,
                "type" to event.type,
                "timestampMs" to event.timestampMs,
                "value" to event.value,
                "category" to event.category
            )
        ).await()
    }

    override suspend fun deleteUser(userId: String) {
        // Implement server-side deletion/batched cleanup according to your data-retention policy.
    }
}
*/
