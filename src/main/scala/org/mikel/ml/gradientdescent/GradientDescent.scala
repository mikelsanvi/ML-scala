package org.mikel.ml.gradientdescent

import scala.annotation.tailrec

/**
  * Created by mikel on 1/06/16.
  */
class GradientDescent(alpha:Double, lambda:Double, trainData: List[InputRow], iterations:Int) {

  private val thetas = iterate(List.fill(trainData.head.features.size)(1.0), iterations)

  @tailrec
  private def iterate(currentThetas:List[Double], iterations:Int): List[Double] = {
    if(iterations == 0) {
      currentThetas
    } else {
      iterate(calculateThetas(currentThetas),iterations -1)
    }
  }

  private def calculateThetas(currentThetas:List[Double]):List[Double] = {
    currentThetas.zip(0 until currentThetas.size).map { case(theta,i) =>
      val summatory = trainData.map(
        row => {
          val prediction = currentThetas.zip(row.features).map{case (theta, x) => theta * x }.sum
          row.features(i) * (prediction - row.y)
        }
      ).sum
      theta* (1.0 - lambda /trainData.size) - (alpha / trainData.size) * summatory
    }
  }

  private def calculateCost(theta:List[Double]): Double = {
    (1 / (2 * trainData.size)) * (trainData.map(
      row => {
        val prediction = theta.zip(row.features).map{case (theta, x) => theta * x }.sum
        Math.pow(prediction - row.y ,2)
      }
    ).sum )
  }

  def predict(input:List[Double]):Double = {
    thetas.zip(input).map{case (theta, x) => theta * x }.sum
  }
}
