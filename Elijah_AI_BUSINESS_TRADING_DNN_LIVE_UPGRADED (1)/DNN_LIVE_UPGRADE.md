# DNN Live Business + Market Upgrade

This upgrade turns the existing small DNN from a standalone chatbot classifier
into a supervised business/market prediction subsystem.

## Training
- Business conversion examples are generated from permitted event histories.
- Market direction examples are generated from historical OHLCV candles using
  future-return labels while keeping future data out of the feature vector.
- Both heads train with backpropagation + Adam.
- Trained weights are persisted in Android SharedPreferences.

## Live inference
- `MarketDataSource` supplies fresh candles.
- `JsonMarketDataSource` can call an application backend.
- `LivePredictionEngine.startMarketPolling()` refreshes data and emits live
  predictions.
- Android `AIViewModel` exposes training, inference and polling methods.

## Safety
This is decision-support/paper-analysis software. It does not execute trades.
Use a server-side credential boundary, validate data, backtest with
walk-forward splits, account for fees/slippage, and impose hard risk limits
before considering any execution integration.
