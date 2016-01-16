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

package org.apache.spark.cuda

import org.apache.commons.io.IOUtils

import org.apache.spark._
import org.apache.spark.rdd._

class CUDACodeGeneratorSuite extends SparkFunSuite with LocalSparkContext {

  private val conf = new SparkConf(false).set("spark.driver.maxResultSize", "2g")
                                         .set("spark.gpu.codegen", "true")

  test("Run map on int rdd with GPU - multiple partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = {
      try {
        new CUDAManager
      } catch {
        case ex: Exception => null
      }
    }
    if (manager != null && manager.deviceCount > 0) {
      val n = 100
      val output = sc.parallelize(1 to n, 2)
        .convert(ColumnFormat)
        .map(x => x + 2)
        .collect()
      assert(output.sameElements((1 to n).map(_ + 2)))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map on double rdd with GPU - multiple partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = {
      try {
        new CUDAManager
      } catch {
        case ex: Exception => null
      }
    }
    if (manager != null && manager.deviceCount > 0) {
      val n = 100
      val output = sc.parallelize((for (i <- 1 to n) yield i.toDouble), 2)
        .convert(ColumnFormat)
        .map(x => x * 1.25)
        .collect()
      assert(output.sameElements((1 to n).map(_ * 1.25)))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run reduce on int rdd with GPU - multiple partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = {
      try {
        new CUDAManager
      } catch {
        case ex: Exception => null
      }
    }
    if (manager != null && manager.deviceCount > 0) {
      val n = 100
      val output = sc.parallelize(1 to n, 2)
        .convert(ColumnFormat)
        .reduce((x, y) => (x + y))
      assert(output == n * (n + 1) / 2)
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run reduce on double rdd with GPU - multiple partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = {
      try {
        new CUDAManager
      } catch {
        case ex: Exception => null
      }
    }
    if (manager != null && manager.deviceCount > 0) {
      val n = 100
      val output = sc.parallelize((for (i <- 1 to n) yield i.toDouble), 2)
        .convert(ColumnFormat)
        .reduce((x, y) => (x + y))
      assert(output == n * (n + 1) / 2)
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + reduce on int rdd with GPU - multiple partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = {
      try {
        new CUDAManager
      } catch {
        case ex: Exception => null
      }
    }
    if (manager != null && manager.deviceCount > 0) {
      val n = 100
      val output = sc.parallelize(1 to n, 2)
        .convert(ColumnFormat)
        .map(x => x * 2)
        .reduce((x, y) => (x + y))
      assert(output == n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

}
