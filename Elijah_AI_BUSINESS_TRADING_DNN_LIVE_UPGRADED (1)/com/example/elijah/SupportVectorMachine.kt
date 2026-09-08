package com.example.elijah

import kotlin.math.max

/** Linear binary SVM trained with hinge-loss SGD. Labels must be -1 or +1. */
class SupportVectorMachine(private val epochs:Int=50, private val learningRate:Double=0.01, private val lambda:Double=0.0001){
    private var w=DoubleArray(0); private var b=0.0
    fun fit(x:Array<DoubleArray>, y:IntArray):SupportVectorMachine{
        require(x.isNotEmpty()&&x.size==y.size&&y.all{it==1||it==-1})
        w=DoubleArray(x[0].size); b=0.0
        repeat(epochs.coerceAtMost(1000)){e->
            val eta=learningRate/(1.0+0.01*e)
            for(i in x.indices){
                val margin=y[i]*(dot(w,x[i])+b)
                for(j in w.indices) w[j]-=eta*(lambda*w[j]-if(margin<1)y[i]*x[i][j] else 0.0)
                if(margin<1) b+=eta*y[i]
            }
        }; return this
    }
    fun predict(x:DoubleArray):Int=if(dot(w,x)+b>=0)1 else -1
    fun score(x:Array<DoubleArray>,y:IntArray):Double=x.indices.count{predict(x[it])==y[it]}.toDouble()/max(1,x.size)
    private fun dot(a:DoubleArray,b:DoubleArray)=a.indices.sumOf{a[it]*b[it]}
}
