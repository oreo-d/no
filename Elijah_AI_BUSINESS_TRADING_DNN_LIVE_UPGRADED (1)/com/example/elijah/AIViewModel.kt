package com.example.elijah

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Upgraded Android ViewModel:
 * NLP -> classifier -> embedding -> vector retrieval -> RAG answer.
 *
 * The trainable classifier remains separate from RAG retrieval. This lets
 * corrections improve classification without corrupting the knowledge index.
 */
class AIViewModel(context: Context) : ViewModel() {

    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences("elijah_ai_v4", Context.MODE_PRIVATE)

    private val textProcessor = TextProcessor()
    private val classifier = MLClassifier()
    private val neuralNetwork = NeuralNetwork()
    private val liveDnnEngine = LivePredictionEngine()

    private val embeddingModel = EmbeddingModel(384)
    private val vectorStore = VectorStore(appContext, embeddingModel)
    private val ragPipeline = RagPipeline(vectorStore)

    @Volatile
    private var realtimeLlm: RealtimeLLM? = null

    @Volatile
    private var realtimeRag: RealtimeRagService? = null

    private val trainingMemory = mutableListOf<TrainingExample>()
    private val businessIntelligence = PrivacyIntelligence()

    private val _messages = MutableStateFlow(
        listOf(ChatMessage("Elijah AI RAG engine is ready. 🤖", false))
    )
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _stats = MutableStateFlow(AIStats())
    val stats: StateFlow<AIStats> = _stats.asStateFlow()

    private var lastFeatures = FloatArray(512)
    private var lastMessage = ""

    init {
        loadModel()
        loadTrainingMemory()
        loadLiveDnn()
        if (vectorStore.size() == 0) {
            vectorStore.upsertAll(KnowledgeBase.defaultDocuments())
        }
        refreshStats()
    }

    fun sendMessage(message: String) {
        val clean = message.trim()
        if (clean.isBlank()) return

        lastMessage = clean
        _messages.update { it + ChatMessage(clean, true) }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val live = realtimeRag
                if (live != null) {
                    val answer = withContext(Dispatchers.IO) {
                        val partial = StringBuilder()
                        val result = live.streamAnswer(clean) { token ->
                            partial.append(token)
                            // UI can observe the assistant message while tokens arrive.
                            _messages.update { list ->
                                if (list.lastOrNull()?.isUser == false)
                                    list.dropLast(1) + ChatMessage(partial.toString(), false)
                                else list + ChatMessage(partial.toString(), false)
                            }
                        }
                        updateRealtimeStats(result.sources.size)
                        result.text + if (result.sources.isNotEmpty()) {
                            "\n\nSources: " + result.sources.take(4).joinToString(", ") { it.source }
                        } else ""
                    }
                    _messages.update { list ->
                        if (list.lastOrNull()?.isUser == false) list.dropLast(1) + ChatMessage(answer, false)
                        else list + ChatMessage(answer, false)
                    }
                } else {
                    val result = withContext(Dispatchers.Default) { processMessage(clean) }
                    _messages.update { it + ChatMessage(result, false) }
                }
            } catch (t: Throwable) {
                _messages.update { it + ChatMessage("I couldn't reach the realtime AI service: ${t.message ?: "unknown error"}. Falling back to local RAG.", false) }
                val fallback = withContext(Dispatchers.Default) { processMessage(clean) }
                _messages.update { it + ChatMessage(fallback, false) }
            } finally {
                _isLoading.value = false
            }
        }
    }

    /** Configure a live OpenAI-compatible model endpoint. Set null key for local servers. */
    fun configureRealtimeLLM(
        endpoint: String,
        model: String,
        apiKey: String? = null
    ) {
        if (endpoint.isBlank() || model.isBlank()) {
            realtimeLlm = null
            realtimeRag = null
            return
        }
        val llm = RealtimeLLM(endpoint, model, apiKey)
        realtimeLlm = llm
        realtimeRag = RealtimeRagService(vectorStore, llm)
    }

    fun disableRealtimeLLM() {
        realtimeLlm = null
        realtimeRag = null
    }

    fun isRealtimeEnabled(): Boolean = realtimeRag != null

    /**
     * Add arbitrary application knowledge. The text is embedded and indexed
     * immediately, so future questions can retrieve it.
     */
    fun addKnowledge(
        id: String,
        text: String,
        source: String = "user"
    ) {
        if (id.isBlank() || text.isBlank()) return
        viewModelScope.launch(Dispatchers.Default) {
            vectorStore.upsert(
                VectorStore.Document(
                    id = id,
                    text = text.trim(),
                    source = source
                )
            )
            refreshStats()
        }
    }

    fun removeKnowledge(id: String) {
        viewModelScope.launch(Dispatchers.Default) {
            vectorStore.remove(id)
            refreshStats()
        }
    }

    fun clearKnowledge() {
        viewModelScope.launch(Dispatchers.Default) {
            vectorStore.clear()
            refreshStats()
        }
    }

    /**
     * Search without generating an answer. Useful for a UI knowledge browser.
     */
    suspend fun searchKnowledge(query: String, topK: Int = 5): List<VectorStore.SearchResult> =
        withContext(Dispatchers.Default) {
            vectorStore.search(query, topK)
        }

    private fun processMessage(message: String): String {
        val features = textProcessor.createFeatures(message)
        lastFeatures = features.copyOf()

        val intent = classifier.classifyDetailed(features, message)
        val prediction = neuralNetwork.predict(features)
        val categoryIndex = prediction.indices.maxByOrNull { prediction[it] } ?: 0
        val confidence = prediction.maxOrNull()?.toDouble() ?: 0.0
        val sentiment = textProcessor.sentimentScore(message)

        val rag = ragPipeline.answer(message, topK = 4)

        _stats.update {
            it.copy(
                totalMessages = it.totalMessages + 1,
                predictions = it.predictions + 1,
                lastConfidence = confidence * 100.0,
                sentiment = sentiment,
                lastIntent = intent.intent.name,
                lastRetrieved = rag.sources.size
            )
        }

        val header = when (intent.intent) {
            MLClassifier.Intent.GREETING ->
                "Hello! 👋 I'm Elijah AI. My local NLP, embeddings and RAG pipeline are ready."
            MLClassifier.Intent.MATH ->
                "🧮 Mathematical input detected. I can route this to a dedicated calculator."
            else ->
                "🧠 Local model: ${categoryName(categoryIndex)} " +
                    "(${String.format("%.1f", confidence * 100)}% confidence)"
        }

        val retrievalInfo = if (rag.usedRetrieval) {
            "\n\n🔎 Retrieved ${rag.sources.size} knowledge chunk(s):\n" +
                rag.sources.joinToString("\n") {
                    "• ${it.source} — ${String.format("%.3f", it.score)}"
                }
        } else {
            "\n\n🔎 No sufficiently similar local knowledge was retrieved."
        }

        return "$header\n\n${rag.text}$retrievalInfo"
    }

    fun trainCorrection(targetCategoryIndex: Int) {
        if (targetCategoryIndex !in 0..2 || lastMessage.isBlank()) return

        trainingMemory += TrainingExample(
            features = lastFeatures.copyOf(),
            targetCategory = targetCategoryIndex
        )

        while (trainingMemory.size > MAX_TRAINING_MEMORY) {
            trainingMemory.removeAt(0)
        }

        saveTrainingMemory()

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val result = withContext(Dispatchers.Default) {
                    neuralNetwork.trainBatch(trainingMemory.map { it.copyExample() }, 20)
                }
                saveModel()
                _stats.update {
                    it.copy(
                        corrections = it.corrections + 1,
                        trainingRuns = it.trainingRuns + 1,
                        trainingExamples = trainingMemory.size,
                        trainingSteps = neuralNetwork.getTrainingSteps(),
                        lastLoss = result.loss,
                        accuracy = result.accuracy
                    )
                }
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun trainAllExamples() {
        if (trainingMemory.isEmpty()) return

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val result = withContext(Dispatchers.Default) {
                    neuralNetwork.trainBatch(trainingMemory.map { it.copyExample() }, 20)
                }
                saveModel()
                _stats.update {
                    it.copy(
                        trainingRuns = it.trainingRuns + 1,
                        trainingExamples = trainingMemory.size,
                        trainingSteps = neuralNetwork.getTrainingSteps(),
                        lastLoss = result.loss,
                        accuracy = result.accuracy
                    )
                }
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun resetLearning() {
        viewModelScope.launch(Dispatchers.Default) {
            neuralNetwork.reset()
            liveDnnEngine.model().reset()
            trainingMemory.clear()
            prefs.edit().clear().apply()
            refreshStats()
            _messages.value = listOf(
                ChatMessage("Local classifier learning was reset. Knowledge documents remain indexed.", false)
            )
        }
    }


    /**
     * Train the business + market DNN on permitted historical data.
     * Market labels are generated from future returns without look-ahead
     * leakage; business labels are generated from future conversions.
     */
    suspend fun trainBusinessMarketDnn(
        userId: String,
        candles: List<MarketCandle>,
        epochs: Int = 25
    ): BusinessMarketDnn.TrainingReport = withContext(Dispatchers.Default) {
        val events = businessEventsFor(userId)
        val report = liveDnnEngine.train(events, candles, epochs)
        saveLiveDnn()
        report
    }

    /**
     * Predict immediately from the newest permitted events and candles.
     * This is inference only and does not place any trade.
     */
    fun predictBusinessMarketLive(
        userId: String,
        candles: List<MarketCandle>
    ): LivePredictionEngine.LiveSnapshot =
        liveDnnEngine.snapshot(businessEventsFor(userId), candles)

    fun liveDnnModel(): BusinessMarketDnn = liveDnnEngine.model()

    /**
     * Start periodic live inference. The provider supplies fresh market data;
     * the callback can update Compose/UI state. No orders are submitted.
     */
    fun startLiveMarketPredictions(
        source: MarketDataSource,
        symbol: String,
        intervalMs: Long = 60_000L,
        userId: String = "",
        onPrediction: (LivePredictionEngine.LiveSnapshot) -> Unit
    ): kotlinx.coroutines.Job =
        liveDnnEngine.startMarketPolling(
            scope = viewModelScope,
            source = source,
            symbol = symbol,
            intervalMs = intervalMs,
            onPrediction = onPrediction,
            businessEvents = { if (userId.isBlank()) emptyList() else businessEventsFor(userId) }
        )

    private fun businessEventsFor(userId: String): List<UserEvent> {
        // PrivacyIntelligence is kept local in this ViewModel. Apps that use
        // Firestore should load only consented events into this list.
        return businessIntelligence.eventsFor(userId)
    }

    private fun saveLiveDnn() {
        runCatching {
            prefs.edit()
                .putString("business_market_dnn_business", liveDnnEngine.model().exportBusinessState())
                .putString("business_market_dnn_market", liveDnnEngine.model().exportMarketState())
                .apply()
        }
    }

    private fun loadLiveDnn() {
        runCatching {
            prefs.getString("business_market_dnn_business", null)?.let {
                liveDnnEngine.model().importBusinessState(it)
            }
            prefs.getString("business_market_dnn_market", null)?.let {
                liveDnnEngine.model().importMarketState(it)
            }
        }
    }

    fun recordBusinessEvent(event: UserEvent) {
        businessIntelligence.record(event)
    }

    fun clearChat() {
        _messages.value = listOf(ChatMessage("New conversation started. 🤖", false))
    }

    fun refreshStats() {
        _stats.update {
            it.copy(
                trainingExamples = trainingMemory.size,
                trainingSteps = neuralNetwork.getTrainingSteps(),
                lastLoss = neuralNetwork.lastLoss,
                accuracy = neuralNetwork.lastAccuracy,
                learningRate = neuralNetwork.currentLearningRate(),
                knowledgeDocuments = vectorStore.size()
            )
        }
    }


    private fun updateRealtimeStats(retrieved: Int) {
        _stats.update {
            it.copy(
                totalMessages = it.totalMessages + 1,
                predictions = it.predictions + 1,
                lastRetrieved = retrieved
            )
        }
    }

    private fun categoryName(index: Int): String = when (index) {
        0 -> "Kotlin / Programming"
        1 -> "Android"
        2 -> "Machine Learning / AI"
        else -> "Unknown"
    }

    private fun saveModel() {
        runCatching {
            prefs.edit().putString("model", neuralNetwork.exportStateJson()).apply()
        }
    }

    private fun loadModel() {
        prefs.getString("model", null)?.let { raw ->
            runCatching { neuralNetwork.importStateJson(raw) }
                .onFailure { neuralNetwork.reset() }
        }
    }

    private fun saveTrainingMemory() {
        runCatching {
            val array = JSONArray()
            trainingMemory.forEach {
                array.put(JSONObject().apply {
                    put("features", JSONArray(it.features.toList()))
                    put("target", it.targetCategory)
                    put("timestamp", it.timestamp)
                })
            }
            prefs.edit().putString("training", array.toString()).apply()
        }
    }

    private fun loadTrainingMemory() {
        val raw = prefs.getString("training", null) ?: return
        runCatching {
            val array = JSONArray(raw)
            trainingMemory.clear()
            for (i in 0 until array.length()) {
                val item = array.getJSONObject(i)
                val f = item.getJSONArray("features")
                trainingMemory += TrainingExample(
                    FloatArray(f.length()) { j -> f.getDouble(j).toFloat() },
                    item.getInt("target"),
                    item.optLong("timestamp", System.currentTimeMillis())
                )
            }
        }.onFailure { trainingMemory.clear() }
    }

    data class ChatMessage(
        val text: String,
        val isUser: Boolean
    )

    data class AIStats(
        val totalMessages: Int = 0,
        val predictions: Int = 0,
        val corrections: Int = 0,
        val trainingRuns: Int = 0,
        val trainingExamples: Int = 0,
        val trainingSteps: Long = 0L,
        val lastConfidence: Double = 0.0,
        val lastLoss: Double = 0.0,
        val accuracy: Double = 0.0,
        val learningRate: Double = 0.003,
        val sentiment: Double = 0.0,
        val lastIntent: String = "UNKNOWN",
        val lastRetrieved: Int = 0,
        val knowledgeDocuments: Int = 0
    )

    companion object {
        private const val MAX_TRAINING_MEMORY = 2000
    }
}

class AIViewModelFactory(
    private val context: Context
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(AIViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return AIViewModel(context.applicationContext) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
