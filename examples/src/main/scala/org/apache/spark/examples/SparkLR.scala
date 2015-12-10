/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples

import java.util.Random

import scala.math.exp

import breeze.linalg.{Vector, DenseVector}

import org.apache.spark._

/**
 * Logistic regression based classification.
 * Usage: SparkLR [slices]
 *
 * This is an example implementation for learning how to use Spark. For more conventional use,
 * please refer to either org.apache.spark.mllib.classification.LogisticRegressionWithSGD or
 * org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS based on your needs.
 */
object SparkLR {
  val rand = new Random(42)

  case class DataPoint(x: Vector[Double], y: Double)

  def generateData(seed: Int, N: Int, D: Int, R: Double): DataPoint = {
    val r = new Random(seed)
    def generatePoint(i: Int): DataPoint = {
      val y = if (i % 2 == 0) -1 else 1
      val x = DenseVector.fill(D){r.nextGaussian + y * R}
      DataPoint(x, y)
    }
    generatePoint(seed)
  }

  def showWarning() {
    System.err.println(
      """WARN: This is a naive implementation of Logistic Regression and is given as an example!
        |Please use either org.apache.spark.mllib.classification.LogisticRegressionWithSGD or
        |org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
        |for more conventional use.
      """.stripMargin)
  }

  def main(args: Array[String]) {

    showWarning()

    val sparkConf = new SparkConf().setAppName("SparkLR")
    val sc = new SparkContext(sparkConf)

    val numSlices = if (args.length > 0) args(0).toInt else 2
    val N = if (args.length > 1) args(1).toInt else 10000  // Number of data points
    val D = if (args.length > 2) args(2).toInt else 10   // Numer of dimensions
    val R = 0.7  // Scaling factor
    val ITERATIONS = if (args.length > 3) args(3).toInt else 5

    val skelton = sc.parallelize((1 to N), numSlices)
    val points = skelton.map(i => generateData(i, N, D, R)).cache()
    points.count()
 
    // Initialize w to a random value
    var w = DenseVector.fill(D){2 * rand.nextDouble - 1}
    printf("numSlices=%d, N=%d, D=%d, ITERATIONS=%d\n", numSlices, N, D, ITERATIONS)
    //println("Initial w: " + w)

    val now = System.nanoTime
    for (i <- 1 to ITERATIONS) {
      println("On iteration " + i)
      val wbc = sc.broadcast(w)
      val gradient = points.map { p =>
        p.x * (1 / (1 + exp(-p.y * (wbc.value.dot(p.x)))) - 1) * p.y
      }.reduce(_ + _)
      w -= gradient
    }
    val ms = (System.nanoTime - now) / 1000000
    println("Elapsed time: %d ms".format(ms))

    //println("Final w: " + w)

    sc.stop()
  }
}
// scalastyle:on println
