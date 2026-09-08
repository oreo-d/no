package com.example.elijah

import kotlin.math.pow
import kotlin.math.sqrt

data class RiskReport(
    val totalReturn: Double,
    val annualizedVolatility: Double,
    val sharpeRatio: Double,
    val maxDrawdown: Double
)

data class TradingSignal(
    val action: String,
    val confidence: Double,
    val reason: String
)

/**
 * Analytics and paper-trading support only. It does not place orders or promise returns.
 * Inputs should be adjusted prices/returns from a trusted market-data source.
 */
class TradingAnalytics(private val periodsPerYear: Double = 252.0) {
    fun returns(prices: List<Double>): List<Double> {
        if (prices.size < 2) return emptyList()
        return prices.zipWithNext().map { (a, b) -> if (a == 0.0) 0.0 else b / a - 1.0 }
    }

    fun riskReport(prices: List<Double>, riskFreeRateAnnual: Double = 0.0): RiskReport {
        val r = returns(prices)
        if (r.isEmpty()) return RiskReport(0.0, 0.0, 0.0, 0.0)
        val mean = r.average()
        val variance = r.map { (it - mean).pow(2) }.average()
        val vol = sqrt(variance) * sqrt(periodsPerYear)
        val annualReturn = (1.0 + mean).pow(periodsPerYear) - 1.0
        val sharpe = if (vol == 0.0) 0.0 else (annualReturn - riskFreeRateAnnual) / vol
        var peak = prices.first()
        var maxDd = 0.0
        prices.forEach { p ->
            peak = maxOf(peak, p)
            maxDd = minOf(maxDd, p / peak - 1.0)
        }
        return RiskReport(
            totalReturn = prices.last() / prices.first() - 1.0,
            annualizedVolatility = vol,
            sharpeRatio = sharpe,
            maxDrawdown = maxDd
        )
    }

    /** Non-executing example signal: moving-average crossover. */
    fun movingAverageSignal(prices: List<Double>, fast: Int = 10, slow: Int = 30): TradingSignal {
        if (prices.size < slow || fast <= 0 || fast >= slow)
            return TradingSignal("HOLD", 0.0, "Insufficient data for the selected windows.")
        val f = prices.takeLast(fast).average()
        val s = prices.takeLast(slow).average()
        val gap = if (s == 0.0) 0.0 else kotlin.math.abs(f - s) / s
        return if (f > s)
            TradingSignal("WATCH / BULLISH", (gap * 10).coerceIn(0.0, 1.0), "Fast moving average is above slow moving average.")
        else
            TradingSignal("WATCH / BEARISH", (gap * 10).coerceIn(0.0, 1.0), "Fast moving average is below slow moving average.")
    }

    /** Simple risk-based position sizing for paper analysis; result is units, not an order. */
    fun positionSize(accountValue: Double, riskFraction: Double, entry: Double, stop: Double): Double {
        val riskPerUnit = kotlin.math.abs(entry - stop)
        if (accountValue <= 0 || riskFraction <= 0 || riskPerUnit <= 0) return 0.0
        return accountValue * riskFraction.coerceIn(0.0, 1.0) / riskPerUnit
    }
}
