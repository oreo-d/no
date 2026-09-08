package com.example.elijah

/** Ordinary least-squares linear regression: y = b0 + b1*x. */
class LinearRegression {
    var slope: Double = 0.0
        private set
    var intercept: Double = 0.0
        private set

    fun fit(x: DoubleArray, y: DoubleArray): LinearRegression {
        require(x.isNotEmpty() && x.size == y.size)
        val mx=x.average(); val my=y.average()
        var num=0.0; var den=0.0
        for(i in x.indices){ val dx=x[i]-mx; num += dx*(y[i]-my); den += dx*dx }
        slope = if (den == 0.0) 0.0 else num/den
        intercept = my-slope*mx
        return this
    }
    fun predict(x: Double): Double = intercept+slope*x
}
