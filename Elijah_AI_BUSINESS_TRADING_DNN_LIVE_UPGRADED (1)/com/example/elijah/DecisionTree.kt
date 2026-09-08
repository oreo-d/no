package com.example.elijah

/** Small numeric decision tree classifier using greedy Gini-split selection. */
class DecisionTree(private val maxDepth:Int=5, private val minSamples:Int=2) {
    private sealed class Node
    private data class Leaf(val label:Int):Node()
    private data class Split(val feature:Int,val threshold:Double,val left:Node,val right:Node):Node()
    private var root:Node?=null

    fun fit(x:Array<DoubleArray>, y:IntArray):DecisionTree {
        require(x.isNotEmpty() && x.size==y.size)
        root=build(x.indices.toList(),x,y,0); return this
    }
    fun predict(row:DoubleArray):Int {
        fun walk(n:Node):Int=when(n){
            is Leaf->n.label
            is Split->walk(if(row[n.feature]<=n.threshold)n.left else n.right)
        }
        return walk(root ?: error("Tree not trained"))
    }
    private fun build(ids:List<Int>,x:Array<DoubleArray>,y:IntArray,depth:Int):Node{
        val labels=ids.map{y[it]}.distinct()
        if(labels.size==1 || depth>=maxDepth || ids.size<minSamples) return Leaf(ids.groupingBy{y[it]}.eachCount().maxBy{it.value}.key)
        var bestGain=0.0; var bf=-1; var bt=0.0
        val base=gini(ids,y)
        for(f in x[0].indices){
            val vals=ids.map{x[it][f]}.distinct().sorted()
            for(i in 0 until vals.lastIndex){
                val t=(vals[i]+vals[i+1])/2.0
                val l=ids.filter{x[it][f]<=t}; val r=ids.filter{x[it][f]>t}
                if(l.isEmpty()||r.isEmpty()) continue
                val gain=base-(l.size*gini(l,y)+r.size*gini(r,y))/ids.size
                if(gain>bestGain){bestGain=gain;bf=f;bt=t}
            }
        }
        if(bf<0) return Leaf(ids.groupingBy{y[it]}.eachCount().maxBy{it.value}.key)
        val l=ids.filter{x[it][bf]<=bt}; val r=ids.filter{x[it][bf]>bt}
        return Split(bf,bt,build(l,x,y,depth+1),build(r,x,y,depth+1))
    }
    private fun gini(ids:List<Int>,y:IntArray):Double{
        val n=ids.size.toDouble(); return 1.0-ids.groupingBy{y[it]}.eachCount().values.sumOf{(it/n)*(it/n)}
    }
}
