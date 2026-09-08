package com.example.elijah

/** UI integration example. Observe AIViewModel.messages from Compose or XML. */
object RealtimeChatExample {
    fun configure(viewModel: AIViewModel) {
        viewModel.configureRealtimeLLM(
            endpoint = "https://YOUR-SERVER/v1/chat/completions",
            model = "YOUR-MODEL",
            apiKey = null
        )
    }
}
