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

import scala.math._

import org.apache.spark._
import org.apache.spark.cuda._

/**
 * Logistic regression based classification.
 * Usage: SparkGPULR [slices] [N] [D] [ITERATION]
 *
 * This is an example implementation for learning how to use Spark. For more conventional use,
 * please refer to either org.apache.spark.mllib.classification.LogisticRegressionWithSGD or
 * org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS based on your needs.
 */
object SparkGPULR {
  val rand = new Random(42)

  case class DataPoint(x: Array[Double], y: Double)

  def dmulvs(x: Array[Double], c: Double) : Array[Double] =
    Array.tabulate(x.length)(i => x(i) * c)
  def daddvv(x: Array[Double], y: Array[Double]) : Array[Double] =
    Array.tabulate(x.length)(i => x(i) + y(i))
  def dsubvv(x: Array[Double], y: Array[Double]) : Array[Double] =
    Array.tabulate(x.length)(i => x(i) - y(i))
  def ddotvv(x: Array[Double], y: Array[Double]) : Double =
    (x zip y).foldLeft(0.0)((a, b) => a + (b._1 * b._2))

  def generateData(seed: Int, N: Int, D: Int, R: Double): DataPoint = {
    val r = new Random(seed)
    def generatePoint(i: Int): DataPoint = {
      val y = if (i % 2 == 0) -1 else 1
      val x = Array.fill(D){r.nextGaussian + y * R}
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

    val sparkConf = new SparkConf().setAppName("SparkGPULR")
    val sc = new SparkContext(sparkConf)

    val numSlices = if (args.length > 0) args(0).toInt else 2
    val N = if (args.length > 1) args(1).toInt else 10000  // Number of data points
    val D = if (args.length > 2) args(2).toInt else 10   // Numer of dimensions
    val R = 0.7  // Scaling factor
    val ITERATIONS = if (args.length > 3) args(3).toInt else 5

    val ptxURL = SparkGPUPi.getClass.getResource("/SparkGPUExamples.ptx")
    val mapFunction = sc.broadcast(
      new CUDAFunction(
      "_Z14SparkGPULR_mapPKlPKdS2_PlPdlS2_",
      Array("this.x", "this.y"),
      Array("this"),
      ptxURL))
    val threads = 1024
    val blocks = min((N + threads- 1) / threads, 1024) 
    val dimensions = (size: Long, stage: Int) => stage match {
      case 0 => (blocks, threads)
    }
    val reduceFunction = sc.broadcast(
      new CUDAFunction(
      "_Z17SparkGPULR_reducePKlPKdPlPdlii",
      Array("this"),
      Array("this"),
      ptxURL,
      Seq(),
      Some((size: Long) => 1),
      Some(dimensions)))

    val skelton = sc.parallelize((1 to N), numSlices)
    val points = skelton.map(i => generateData(i, N, D, R))
    val pointsColumnCached = points.convert(ColumnFormat).cache().cacheGpu()
    pointsColumnCached.count()

    // Initialize w to a random value
    var w = Array.fill(D){2 * rand.nextDouble - 1}
    printf("numSlices=%d, N=%d, D=%d, ITERATIONS=%d\n", numSlices, N, D, ITERATIONS)
    //println("Initial w: " + w)

    val now = System.nanoTime
    for (i <- 1 to ITERATIONS) {
      println("On iteration " + i)
      val wbc = sc.broadcast(w)
      val gradient = pointsColumnCached.mapExtFunc((p: DataPoint) =>
        dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wbc.value, p.x)))) - 1) * p.y),
        mapFunction.value, outputArraySizes = Array(D),
        inputFreeVariables = Array(wbc.value)
      ).reduceExtFunc((x: Array[Double], y: Array[Double]) => daddvv(x, y),
                      reduceFunction.value, outputArraySizes = Array(D))
      w = dsubvv(w, gradient)
    }
    val ms = (System.nanoTime - now) / 1000000
    println("Elapsed time: %d ms".format(ms))

    pointsColumnCached.unCacheGpu()

    //println("Final w: " + w)

    sc.stop()
  }
}
// scalastyle:on println
