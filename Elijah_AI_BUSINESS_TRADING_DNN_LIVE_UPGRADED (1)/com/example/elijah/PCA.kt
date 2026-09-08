package com.example.elijah

import kotlin.math.sqrt

/** PCA via power iteration; useful for dimensionality reduction on small datasets. */
class PCA(private val components:Int){
    var mean=DoubleArray(0); private set
    var vectors=Array(0){DoubleArray(0)}; private set

    fun fit(data:Array<DoubleArray>):PCA{
        require(data.isNotEmpty()&&components>0&&components<=data[0].size)
        mean=DoubleArray(data[0].size){j->data.map{it[j]}.average()}
        val z=Array(data.size){i->DoubleArray(data[i].size){j->data[i][j]-mean[j]}}
        val cov=Array(mean.size){i->DoubleArray(mean.size){j->z.sumOf{it[i]*it[j]}/(z.size-1).coerceAtLeast(1)}}
        val found=mutableListOf<DoubleArray>()
        repeat(components){
            var v=DoubleArray(mean.size){1.0/sqrt(mean.size.toDouble())}
            repeat(80){val nv=DoubleArray(v.size){i->cov[i].indices.sumOf{cov[i][it]*v[it]}};val n=sqrt(nv.sumOf{it*it});if(n>1e-12)for(i in v.indices)v[i]=nv[i]/n}
            found+=v
            val lambda=v.indices.sumOf{i->v[i]*v.indices.sumOf{j->cov[i][j]*v[j]}}
            for(i in cov.indices)for(j in cov.indices)cov[i][j]-=lambda*v[i]*v[j]
        }
        vectors=found.toTypedArray();return this
    }
    fun transform(row:DoubleArray):DoubleArray{require(vectors.isNotEmpty());val c=DoubleArray(row.size){row[it]-mean[it]};return vectors.map{v->v.indices.sumOf{v[it]*c[it]}}.toDoubleArray()}
}
