package com.example.elijah

import kotlin.math.exp
import kotlin.math.sqrt

/** Dependency-free scaled dot-product self-attention encoder block.
 * This is an educational/inference building block, not a pretrained LLM.
 */
class TransformerEncoder(private val dimension:Int){
    fun selfAttention(tokens:Array<FloatArray>):Array<FloatArray>{
        require(tokens.all{it.size==dimension})
        val out=Array(tokens.size){FloatArray(dimension)}
        for(i in tokens.indices){
            val scores=DoubleArray(tokens.size){j->dot(tokens[i],tokens[j])/sqrt(dimension.toDouble())}
            val m=scores.maxOrNull()?:0.0; var sum=0.0
            for(j in scores.indices){scores[j]=exp((scores[j]-m).coerceIn(-40.0,40.0));sum+=scores[j]}
            for(j in tokens.indices)for(d in 0 until dimension)out[i][d]=(out[i][d]+(scores[j]/sum)*tokens[j][d]).toFloat()
        };return out
    }
    private fun dot(a:FloatArray,b:FloatArray)=a.indices.sumOf{a[it].toDouble()*b[it]}
}
