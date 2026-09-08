# Elijah AI — realtime chatbot setup

This package now supports a real-time, token-streaming chatbot while keeping the offline NLP/RAG fallback.

## 1. Android permission

Add this to `app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

## 2. Copy Kotlin files

Copy the `com/example/elijah/*.kt` files into:

`app/src/main/java/com/example/elijah/`

## 3. Configure a model

The app accepts any server exposing an OpenAI-compatible endpoint:

`POST <endpoint>` with `model`, `messages`, `temperature`, `max_tokens`, and `stream`.

Example:

```kotlin
viewModel.configureRealtimeLLM(
    endpoint = "https://YOUR-SERVER/v1/chat/completions",
    model = "YOUR-MODEL",
    apiKey = "YOUR-KEY"
)
```

For a local/LAN model server, omit the API key:

```kotlin
viewModel.configureRealtimeLLM(
    endpoint = "http://192.168.1.50:11434/v1/chat/completions",
    model = "your-local-model"
)
```

Do not hard-code production secrets in a public APK. Prefer a backend proxy or secure Android secret storage.

## 4. Runtime flow

User message -> classifier/NLP -> 384D embedding -> cosine vector retrieval -> grounded prompt -> streaming LLM -> live UI message updates.

If the server is unavailable, the ViewModel automatically falls back to the local extractive RAG pipeline.

## 5. Algorithms included

- Linear Regression
- Decision Tree
- Linear SVM
- K-Means
- PCA
- Feed-forward Neural Network + Adam
- Transformer self-attention building block
- Q-Learning
- NLP embeddings
- Vector search
- RAG
- Streaming LLM client
