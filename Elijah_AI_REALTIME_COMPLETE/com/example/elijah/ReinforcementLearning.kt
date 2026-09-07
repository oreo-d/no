package com.example.elijah

import kotlin.math.max
import kotlin.random.Random

/** Tabular Q-learning engine for small discrete environments. */
class QLearning(private val states:Int, private val actions:Int, private val alpha:Double=.1, private val gamma:Double=.95, private val epsilon:Double=.15){
    private val q=Array(states){DoubleArray(actions)}
    fun chooseAction(state:Int, explore:Boolean=true):Int{
        require(state in 0 until states)
        if(explore && Random.nextDouble()<epsilon)return Random.nextInt(actions)
        return q[state].indices.maxBy{q[state][it]}
    }
    fun update(state:Int,action:Int,reward:Double,nextState:Int,done:Boolean=false):Double{
        val target=reward+if(done)0.0 else gamma*q[nextState].maxOrNull()!!
        val td=target-q[state][action];q[state][action]+=alpha*td;return td
    }
    fun qValues(state:Int)=q[state].copyOf()
}
