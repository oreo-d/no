package com.example.elijah

import kotlin.math.sqrt

/** K-Means clustering with deterministic first-k initialization. */
class KMeans(private val k:Int, private val maxIterations:Int=100){
    var centroids:Array<DoubleArray> = emptyArray(); private set
    fun fit(data:Array<DoubleArray>):IntArray{
        require(k>0&&data.size>=k)
        centroids=Array(k){data[it].copyOf()}; var labels=IntArray(data.size)
        repeat(maxIterations){
            var changed=false
            for(i in data.indices){val c=nearest(data[i]);if(labels[i]!=c||it==0){labels[i]=c;changed=true}}
            val sums=Array(k){DoubleArray(data[0].size)}; val counts=IntArray(k)
            for(i in data.indices){val c=labels[i];counts[c]++;for(j in data[i].indices)sums[c][j]+=data[i][j]}
            for(c in 0 until k)if(counts[c]>0)for(j in sums[c].indices)centroids[c][j]=sums[c][j]/counts[c]
            if(!changed)return labels
        };return labels
    }
    fun predict(point:DoubleArray)=nearest(point)
    private fun nearest(p:DoubleArray)=centroids.indices.minBy{distance(p,centroids[it])}
    private fun distance(a:DoubleArray,b:DoubleArray)=sqrt(a.indices.sumOf{val d=a[it]-b[it];d*d})
}
