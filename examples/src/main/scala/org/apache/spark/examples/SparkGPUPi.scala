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

import scala.math.random

import org.apache.spark._
import org.apache.spark.cuda._

/**
 *  Computes an approximation to pi
 *  This example uses GPU to execute map() and reduce()
 */
object SparkGPUPi {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("SparkGPUPi")
    val spark = new SparkContext(sparkConf)
    val ptxURL = SparkGPUPi.getClass.getResource("/SparkGPUExamples.ptx")
    val mapFunction = new CUDAFunction(
      "_Z14SparkGPUPi_mapPKiPil",
      Array("this"),
      Array("this"),
      ptxURL)

    val dimensions = (size: Long, stage: Int) => stage match {
      case 0 => (64, 256)
      case 1 => (1, 1)
    }
    val reduceFunction = new CUDAFunction(
      "_Z17SparkGPUPi_reducePiS_lii",
      Array("this"),
      Array("this"),
      ptxURL,
      Seq(),
      Some((size: Long) => 2),
      Some(dimensions))

    val slices = if (args.length > 0) args(0).toInt else 2
    val n = 100000 * slices

    val rdd = spark.parallelize(1 to n, slices)
    val count = rdd
      .convert(ColumnFormat)
      .mapExtFunc( (i : Int) => {
        val x = random * 2 - 1
        val y = random * 2 - 1
        if (x * x + y * y < 1) 1 else 0 } ,  
        mapFunction)
      .reduceExtFunc((x: Int, y: Int) => x + y, reduceFunction)
    println("Pi is roughly " + 4.0 * count / n)

    spark.stop()
  }
}
// scalastyle:on println
