# Elijah AI — Embeddings + Vector Search + RAG + Android AIViewModel

This is an upgraded offline Android AI package built on the supplied Elijah AI
classifier/neural-network project.

## Pipeline

User message
  -> TextProcessor
  -> Neural classifier
  -> EmbeddingModel (384 dimensions)
  -> VectorStore cosine search
  -> top-K knowledge chunks
  -> RagPipeline
  -> answer + sources

## New components

### EmbeddingModel.kt
- 384-dimensional dense vectors
- word hashing
- word bigrams
- character trigrams
- positional hashing
- L2 normalization
- cosine similarity
- no external NLP dependency

Important: this is a deterministic local embedding model, not a pretrained
Transformer/Sentence-BERT embedding. It is intended for an offline mobile
retrieval layer.

### VectorStore.kt
- persistent local knowledge index
- document upsert/remove
- batch indexing
- cosine vector search
- metadata/source support
- SharedPreferences persistence for a small corpus

For thousands/millions of chunks, replace it with Room/SQLite + an ANN index.

### RagPipeline.kt
- query embedding
- top-K retrieval
- similarity threshold
- context selection
- pluggable Generator interface
- offline extractive fallback generator

To connect a local LLM later, implement RagPipeline.Generator and pass it
to RagPipeline.

### KnowledgeBase.kt
Includes starter Android/Kotlin/RAG/embedding/ML documents.

### AIViewModel.kt
Adds:
- RAG-aware chat
- addKnowledge()
- removeKnowledge()
- clearKnowledge()
- searchKnowledge()
- retrieval statistics
- existing neural correction/training
- persistent model/training memory

## Android dependencies

The new files intentionally use Android SDK JSON/SharedPreferences and the
same AndroidX lifecycle/coroutines stack already expected by the original
project.

Typical dependencies:

implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:<version>")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:<version>")

Use versions compatible with your existing Gradle setup.

## Placement

Put all Kotlin files under:

app/src/main/java/com/example/elijah/

Files:
- AIViewModel.kt
- EmbeddingModel.kt
- VectorStore.kt
- RagPipeline.kt
- KnowledgeBase.kt
- TrainingExample.kt
- NeuralNetwork.kt
- MLClassifier.kt
- TextProcessor.kt

## Example

val vm = ViewModelProvider(this, AIViewModelFactory(this))
    .get(AIViewModel::class.java)

Add application knowledge:

vm.addKnowledge(
    id = "my_faq_1",
    text = "Your company FAQ or product documentation...",
    source = "Company FAQ"
)

Then send:

vm.sendMessage("How does our product work?")

The ViewModel retrieves matching chunks before producing the answer.

## Next production upgrade

For a stronger semantic stack:
1. Replace EmbeddingModel with a quantized MiniLM/E5/BGE embedding model.
2. Store vectors in Room/SQLite or a native ANN index.
3. Chunk PDFs/web pages into 300–800 token passages.
4. Add reranking after vector retrieval.
5. Connect a local GGUF/MediaPipe/LiteRT LLM as the Generator.
6. Add streaming token output.
7. Add citations to individual retrieved chunks.


# Advanced ML algorithms added

This upgrade adds dependency-free Kotlin implementations:

## Supervised learning
- `LinearRegression.kt` — ordinary least-squares continuous prediction.
- `DecisionTree.kt` — greedy Gini-based numeric decision-tree classifier.
- `SupportVectorMachine.kt` — linear binary SVM with hinge-loss SGD.

## Unsupervised learning
- `KMeans.kt` — centroid-based clustering.
- `PCA.kt` — principal-component dimensionality reduction using covariance + power iteration.

## Deep learning / advanced AI
- `TransformerEncoder.kt` — scaled dot-product self-attention building block.
- Existing `NeuralNetwork.kt` — on-device feed-forward classifier with ReLU, Softmax, Adam and replay training.
- `ReinforcementLearning.kt` — tabular Q-learning engine.
- `LocalLLM.kt` — local-LLM abstraction plus a RAG adapter.

### Important
The Transformer and LocalLLM classes are **architectural/inference building blocks**, not a pretrained ChatGPT/Gemini-sized model. A genuine generative local model still needs a model asset (for example a compatible TFLite/MediaPipe or another Android-supported runtime). The `LocalLLM` interface is intentionally isolated so such a model can be connected without changing the vector store or RAG pipeline.

### Example
```kotlin
val regression = LinearRegression().fit(
    doubleArrayOf(1.0, 2.0, 3.0),
    doubleArrayOf(2.0, 4.0, 6.0)
)
val prediction = regression.predict(4.0)

val clusters = KMeans(2).fit(
    arrayOf(
        doubleArrayOf(1.0, 1.0),
        doubleArrayOf(1.2, 0.9),
        doubleArrayOf(8.0, 8.0),
        doubleArrayOf(8.2, 7.9)
    )
)

val ragGenerator = LocalLLMRagGenerator(yourLocalLLM)
val rag = RagPipeline(vectorStore, ragGenerator)
```

## Realtime chatbot upgrade

`RealtimeLLM.kt` and `RealtimeRagService.kt` add real token streaming through an OpenAI-compatible chat-completions endpoint. `AIViewModel.sendMessage()` automatically uses the live pipeline when configured and otherwise remains fully offline.

Configure it with:

```kotlin
viewModel.configureRealtimeLLM(
    endpoint = "https://YOUR-SERVER/v1/chat/completions",
    model = "YOUR-MODEL",
    apiKey = "YOUR-KEY"
)
```

The model endpoint can be a cloud provider, a private server, or a local/LAN inference server that implements the OpenAI-compatible API shape.

See `ANDROID_REALTIME_SETUP.md`.


# Business + Trading Intelligence Upgrade

This package adds a privacy-conscious intelligence layer suitable for business applications and trading analytics.

## Business modules

- `PrivacyIntelligence.kt` — explicit consent gates for analytics, personalization and marketing events.
- `PreferenceEngine` — converts permitted events into simple non-sensitive category scores.
- `BusinessIntelligence.kt` — campaign CTR, conversion rate, ROAS and RFM-style customer segmentation.
- `RecommendationEngine` — ranks catalog/content candidates from preference scores.
- `FirebaseBusinessAdapter.kt` — dependency-free interface for connecting Firestore/Firebase Analytics in the Android app module.

Example events:
`product_viewed`, `product_searched`, `recommendation_clicked`, `ad_impression`, `ad_click`, `campaign_conversion`.

Do not collect passwords, payment-card data, government IDs or other sensitive data in this event layer. Add retention/deletion controls appropriate to your jurisdiction and business.

## Trading analytics

`TradingAnalytics.kt` provides:
- historical returns
- annualized volatility
- Sharpe ratio
- maximum drawdown
- moving-average watch signals
- risk-based position-size calculations for analysis/paper trading

**Important:** these functions are analytics and educational tooling. They do not execute trades and do not guarantee profits. Connect them to a vetted market-data provider and add validation, paper trading and human approval before any production execution system.

## Firebase architecture

Recommended production flow:

App -> Consent UI -> event collector -> Firestore/Firebase Analytics -> Cloud Functions/Cloud Run -> feature aggregation -> recommendation/business dashboards.

For trading:

Market data -> validation -> feature calculations -> model/signal -> risk limits -> paper-trading simulator -> human-approved execution layer.

Keep exchange/broker credentials server-side; never ship them in an Android APK.

## New SVG

`elijah_ai_intelligence.svg` is an architecture artwork that can be used in documentation or the app's project assets.

## Build notes

The new Kotlin files use only Kotlin/JVM/Android-compatible standard APIs and can be copied into the same package:
`app/src/main/java/com/example/elijah/`

Firebase implementation is intentionally commented out so the core ZIP does not force Firebase dependencies. Add Firebase dependencies in the actual Android application module when needed.
