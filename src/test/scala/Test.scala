/**
  * Created by mikel on 1/06/16.
  */

import org.mikel.ml.gradientdescent.{GradientDescent, InputRow}
import org.scalatest.FlatSpec

import scala.util.Random

class Test extends FlatSpec {

  val magnitude = 10000
  val error = 0.01 * magnitude

  "A GD" should "predict correctly" in {

    val input:List[InputRow] = stream(100).take(100).toList

    val gd = new GradientDescent(0.001,0.00, input, 500)

    stream(magnitude).take(100000).foreach(checkPrediction(gd,_))
  }

  def checkPrediction(gd:GradientDescent, inputRow: InputRow) = {
    val prediction = gd.predict(inputRow.features)
    assert(prediction > (inputRow.y - error) && prediction < ( inputRow.y + error ))
  }

  val r = new Random()

  def stream(magnitude:Int): Stream[InputRow] = generateRandomValue(magnitude) #:: stream(magnitude)

  def generateRandomValue(magnitude:Int):InputRow = {
    val x1 = r.nextDouble() * magnitude
    val x2 = r.nextDouble() * magnitude
    val x3 = r.nextDouble() * magnitude
    val error = r.nextDouble()

    new InputRow(List(x1.toDouble,x2.toDouble, x3.toDouble), (error+ 2 * x1 + x2))
  }

}

