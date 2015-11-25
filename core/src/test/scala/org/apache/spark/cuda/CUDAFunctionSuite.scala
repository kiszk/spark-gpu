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

import java.util.{Calendar, Random}

import org.apache.commons.io.IOUtils

import org.apache.spark._

import scala.math._

import scala.reflect.ClassTag

case class Vector2DDouble(x: Double, y: Double)
case class VectorLength(len: Double)
case class PlusMinus(base: Double, deviation: Float)
case class FloatRange(a: Double, b: Float)
case class IntDataPoint(x: Array[Int], y: Int)
case class DataPoint(x: Array[Double], y: Double)

class CUDAFunctionSuite extends SparkFunSuite with LocalSparkContext {

  private val conf = new SparkConf(false).set("spark.driver.maxResultSize", "2g")

  test("Ensure CUDA kernel is serializable", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
    val function = new CUDAFunction(
      "_Z8identityPKiPil",
      Array("this"),
      Array("this"),
      ptxURL)
    SparkEnv.get.closureSerializer.newInstance().serialize(function)
  }

  test("Run identity CUDA kernel on a single primitive column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z8identityPKiPil",
        Array("this"),
        Array("this"),
        ptxURL)
      val n = 1024
      val input = ColumnPartitionDataBuilder.build(1 to n)
      val output = function.run[Int, Int](input)
      assert(output.size == n)
      assert(output.schema.isPrimitive)
      val outputItr  = output.iterator
      assert(outputItr.toIndexedSeq.sameElements(1 to n))
      assert(!outputItr.hasNext)
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run identity CUDA kernel on a single primitive array column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z16intArrayIdentityPKlPKiPlPil",
        Array("this"),
        Array("this"),
        ptxURL)
      val n = 16
      val input = ColumnPartitionDataBuilder.
                    build(Array(Array.range(0, n), Array.range(-(n-1), 1)))
      val output = function.run[Array[Int], Array[Int]](input, outputArraySizes = Array(n))
      assert(output.size == 2)
      assert(output.schema.isPrimitive)
      val outputItr = output.iterator
      assert(outputItr.next.toIndexedSeq.sameElements(0 to n-1))
      assert(outputItr.next.toIndexedSeq.sameElements(-(n-1) to 0))
      assert(!outputItr.hasNext)
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run identity CUDA kernel on a single primitive array in a structure", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z20IntDataPointIdentityPKlPKiS2_PlPiS4_l",
        Array("this.x", "this.y"),
        Array("this.x", "this.y"),
        ptxURL)
      val n = 5
      val dataset = List(IntDataPoint(Array(  1,   2,   3,   4,   5), -10),
                         IntDataPoint(Array( -5,  -4,  -3,  -2,  -1),  10))
      val input = ColumnPartitionDataBuilder.build(dataset)
      val output = function.run[IntDataPoint, IntDataPoint](input, outputArraySizes = Array(n))
      assert(output.size == 2)
      assert(!output.schema.isPrimitive)
      val outputItr = output.iterator
      val next1 = outputItr.next
      assert(next1.x.toIndexedSeq.sameElements(1 to n))
      assert(next1.y == -10)
      val next2 = outputItr.next
      assert(next2.x.toIndexedSeq.sameElements(-n to -1))
      assert(next2.y ==  10)
      assert(!outputItr.hasNext)
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run add CUDA kernel with free variables on a single primitive array column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z11intArrayAddPKlPKiPlPilS2_",
        Array("this"),
        Array("this"),
        ptxURL)
      val n = 16
      val v = Array.fill(n)(1)
      val input = ColumnPartitionDataBuilder.
                    build(Array(Array.range(0, n), Array.range(-(n-1), 1)))
      val output = function.run[Array[Int], Array[Int]](input,
                     outputArraySizes = Array(n),
                     inputFreeVariables = Array(v))
      assert(output.size == 2)
      assert(output.schema.isPrimitive)
      val outputItr = output.iterator
      assert(outputItr.next.toIndexedSeq.sameElements(1 to n))
      assert(outputItr.next.toIndexedSeq.sameElements(-(n-2) to 1))
      assert(!outputItr.hasNext)
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run vectorLength CUDA kernel on 2 col -> 1 col", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z12vectorLengthPKdS0_Pdl",
        Array("this.x", "this.y"),
        Array("this.len"),
        ptxURL)
      val n = 100
      val inputVals = (1 to n).flatMap { x =>
        (1 to n).map { y =>
          Vector2DDouble(x, y)
        }
      }
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = function.run[Vector2DDouble, VectorLength](input)
      assert(output.size == n * n)
      output.iterator.zip(inputVals.iterator).foreach { case (res, vect) =>
        assert(abs(res.len - sqrt(vect.x * vect.x + vect.y * vect.y)) < 1e-7)
      }
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run plusMinus CUDA kernel on 2 col -> 2 col", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z9plusMinusPKdPKfPdPfl",
        Array("this.base", "this.deviation"),
        Array("this.a", "this.b"),
        ptxURL)
      val n = 100
      val inputVals = (1 to n).flatMap { base =>
        (1 to n).map { deviation =>
          PlusMinus(base * 0.1, deviation * 0.01f)
        }
      }
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = function.run[PlusMinus, FloatRange](input)
      assert(output.size == n * n)
      output.iterator.toIndexedSeq.zip(inputVals).foreach { case (range, plusMinus) =>
        assert(abs(range.b - range.a - 2 * plusMinus.deviation) < 1e-5f)
      }
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run applyLinearFunction CUDA kernel on 1 col + 2 const arg -> 1 col", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z19applyLinearFunctionPKsPslss",
        Array("this"),
        Array("this"),
        ptxURL,
        List(2: Short, 3: Short))
      val n = 1000
      val input = ColumnPartitionDataBuilder.build((1 to n).map(_.toShort))
      val output = function.run[Short, Short](input)
      assert(output.size == n)
      output.iterator.toIndexedSeq.zip((1 to n).map(x => (2 + 3 * x).toShort)).foreach {
        case (got, expected) => assert(got == expected)
      }
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run blockXOR CUDA kernel on 1 col + 1 const arg -> 1 col on custom dimensions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      // we only use size/8 GPU threads and run block on a single warp
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val function = new CUDAFunction(
        "_Z8blockXORPKcPcll",
        Array("this"),
        Array("this"),
        ptxURL,
        List(0x0102030411121314l),
        None,
        Some((size: Long, stage: Int) => (((size + 32 * 8 - 1) / (32 * 8)).toInt, 32)))
      val n = 10
      val inputVals = List.fill(n)(List(
          0x14, 0x13, 0x12, 0x11, 0x04, 0x03, 0x02, 0x01,
          0x34, 0x33, 0x32, 0x31, 0x00, 0x00, 0x00, 0x00
        ).map(_.toByte)).flatten
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = function.run[Byte, Byte](input)
      assert(output.size == 16 * n)
      val expectedOutputVals = List.fill(n)(List(
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x20, 0x20, 0x20, 0x20, 0x04, 0x03, 0x02, 0x01
        ).map(_.toByte)).flatten
      assert(output.iterator.toIndexedSeq.sameElements(expectedOutputVals))
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run sum CUDA kernel on 1 col -> 1 col in 2 stages", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val function = new CUDAFunction(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))
      val n = 30000
      val input = ColumnPartitionDataBuilder.build(1 to n)
      val output = function.run[Int, Int](input, Some(1))
      assert(output.size == 1)
      assert(output.iterator.next == n * (n + 1) / 2)
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map on rdds - single partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        ptxURL)

      val n = 10
      val output = sc.parallelize(1 to n, 1)
        .convert(ColumnFormat)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .collect()
      assert(output.sameElements((1 to n).map(_ * 2)))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run reduce on rdds - single partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = new CUDAFunction(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 10
      val output = sc.parallelize(1 to n, 1)
        .convert(ColumnFormat)
        .reduceExtFunc((x: Int, y: Int) => x + y, reduceFunction)
      assert(output == n * (n + 1) / 2)
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + reduce on rdds - single partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        ptxURL)

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = new CUDAFunction(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 10
      val output = sc.parallelize(1 to n, 1)
        .convert(ColumnFormat)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .reduceExtFunc((x: Int, y: Int) => x + y, reduceFunction)
      assert(output == n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map on rdds with 100,000,000 elements - multiple partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        ptxURL)

      val n = 100000000
      val output = sc.parallelize(1 to n, 64)
        .convert(ColumnFormat)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .collect()
      assert(output.sameElements((1 to n).map(_ * 2)))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + reduce on rdds - multiple partitions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        ptxURL)

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = new CUDAFunction(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100
      val output = sc.parallelize(1 to n, 16)
        .convert(ColumnFormat)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .reduceExtFunc((x: Int, y: Int) => x + y, reduceFunction)
      assert(output == n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + reduce on rdds with 100,000,000 elements - multiple partitions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        ptxURL)

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = new CUDAFunction(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100000000
      val output = sc.parallelize(1 to n, 64)
        .convert(ColumnFormat)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .reduceExtFunc((x: Int, y: Int) => x + y, reduceFunction)
      assert(output == n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + map + reduce on rdds - multiple partitions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        ptxURL)

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = new CUDAFunction(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100
      val output = sc.parallelize(1 to n, 16)
        .convert(ColumnFormat)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .reduceExtFunc((x: Int, y: Int) => x + y, reduceFunction)
      assert(output == 2 * n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + map + map + reduce on rdds - multiple partitions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        ptxURL)

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = new CUDAFunction(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100
      val output = sc.parallelize(1 to n, 16)
        .convert(ColumnFormat)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .mapExtFunc((x: Int) => 2 * x, mapFunction)
        .reduceExtFunc((x: Int, y: Int) => x + y, reduceFunction)
      assert(output == 4 * n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map on rdd with a single primitive array column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z16intArrayIdentityPKlPKiPlPil",
        Array("this"),
        Array("this"),
        ptxURL)
      val n = 16
      val dataset = List(Array.range(0, n), Array.range(-(n-1), 1))
      val output = sc.parallelize(dataset, 1)
        .convert(ColumnFormat)
        .mapExtFunc((x: Array[Int]) => x, mapFunction, outputArraySizes = Array(n))
        .collect()
      val outputItr = output.iterator
      assert(outputItr.next.toIndexedSeq.sameElements(0 to n-1))
      assert(outputItr.next.toIndexedSeq.sameElements(-(n-1) to 0))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map with free variables on rdd with a single primitive array column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      def iaddvv(x: Array[Int], y: Array[Int]) : Array[Int] =
        Array.tabulate(x.length)(i => x(i) + y(i))

      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = new CUDAFunction(
        "_Z11intArrayAddPKlPKiPlPilS2_",
        Array("this"),
        Array("this"),
        ptxURL)
      val n = 16
      val v = Array.fill(n)(1)
      val dataset = List(Array.range(0, n), Array.range(-(n-1), 1))
      val output = sc.parallelize(dataset, 1)
        .convert(ColumnFormat)
        .mapExtFunc((x: Array[Int]) => iaddvv(x, v),
                    mapFunction, outputArraySizes = Array(n),
                    inputFreeVariables = Array(v))
        .collect()
      val outputItr = output.iterator
      assert(outputItr.next.toIndexedSeq.sameElements(1 to n))
      assert(outputItr.next.toIndexedSeq.sameElements(-(n-2) to 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run reduce on rdd with a single primitive array column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      def iaddvv(x: Array[Int], y: Array[Int]) : Array[Int] =
        Array.tabulate(x.length)(i => x(i) + y(i))

      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = new CUDAFunction(
        "_Z11intArraySumPKlPKiPlPilii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 8
      val dataset = List(Array.range(0, n), Array.range(2*n, 3*n))
      val output = sc.parallelize(dataset, 1)
        .convert(ColumnFormat)
        .reduceExtFunc((x: Array[Int], y: Array[Int]) => iaddvv(x, y),
                       reduceFunction, outputArraySizes = Array(n))
      assert(output.toIndexedSeq.sameElements((n to 2*n-1).map(_ * 2)))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map & reduce on a single primitive array in a structure", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      def daddvv(x: Array[Double], y: Array[Double]) : Array[Double] = {
        Array.tabulate(x.length)(i => x(i) + y(i))
      }

      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = sc.broadcast(
        new CUDAFunction(
        "_Z12DataPointMapPKlPKiPKdPlPdlS4_",
        Array("this.x", "this.y"),
        Array("this"),
        ptxURL))
      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceFunction = sc.broadcast(
        new CUDAFunction(
        "_Z15DataPointReducePKlPKdPlPdlii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions)))
      val n = 5
      val w = Array.fill(n)(2.0)
      val dataset = List(DataPoint(Array(  1.0,   2.0,   3.0,   4.0,   5.0), -1),
                         DataPoint(Array( -5.0,  -4.0,  -3.0,  -2.0,  -1.0),  1))
      val input = sc.parallelize(dataset, 2).convert(ColumnFormat).cache()
      val output = input.mapExtFunc((p: DataPoint) => daddvv(p.x, w),
                                    mapFunction.value, outputArraySizes = Array(n),
                                    inputFreeVariables = Array(w))
                        .reduceExtFunc((x: Array[Double], y: Array[Double]) => daddvv(x, y),
                                       reduceFunction.value, outputArraySizes = Array(n))
      assert(output.sameElements((0 to 4).map((x: Int) => x * 2.0)))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run logistic regression", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      def dmulvs(x: Array[Double], c: Double) : Array[Double] =
        Array.tabulate(x.length)(i => x(i) * c)
      def daddvv(x: Array[Double], y: Array[Double]) : Array[Double] =
        Array.tabulate(x.length)(i => x(i) + y(i))
      def dsubvv(x: Array[Double], y: Array[Double]) : Array[Double] =
        Array.tabulate(x.length)(i => x(i) - y(i))
      def ddotvv(x: Array[Double], y: Array[Double]) : Double =
        (x zip y).foldLeft(0.0)((a, b) => a + (b._1 * b._2))

      val N = 8192  // Number of data points
      val D = 8   // Numer of dimensions
      val R = 0.7  // Scaling factor
      val ITERATIONS = 5
      val numSlices = 8
      val rand = new Random(42)

      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = sc.broadcast(
        new CUDAFunction(
        "_Z5LRMapPKlPKdS2_PlPdlS2_",
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
        "_Z8LRReducePKlPKdPlPdlii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 1),
        Some(dimensions)))

      def generateData: Array[DataPoint] = {
        def generatePoint(i: Int): DataPoint = {
          val y = if (i % 2 == 0) -1 else 1
          val x = Array.fill(D){rand.nextGaussian + y * R}
          DataPoint(x, y)
        }
        Array.tabulate(N)(generatePoint)
      }

      val points = sc.parallelize(generateData, numSlices)
      points.cacheGpu()
      val pointsColumnCached = points.convert(ColumnFormat).cache()
      val pointsCached = points.cache()

      // Initialize w to a random value
      var wCPU = Array.fill(D){2 * rand.nextDouble - 1}
      var wGPU = Array.tabulate(D)(i => wCPU(i))

      for (i <- 1 to ITERATIONS) {
        val wGPUbcast = sc.broadcast(wGPU)
        val gradient = pointsColumnCached.mapExtFunc((p: DataPoint) =>
          dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wGPU, p.x)))) - 1) * p.y),
          mapFunction.value, outputArraySizes = Array(D),
          inputFreeVariables = Array(wGPUbcast.value)
        ).reduceExtFunc((x: Array[Double], y: Array[Double]) => daddvv(x, y),
          reduceFunction.value, outputArraySizes = Array(D))
        wGPU = dsubvv(wGPU, gradient)
      }
      pointsColumnCached.unCacheGpu()

      for (i <- 1 to ITERATIONS) {
        val gradient = pointsCached.map { p =>
          dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wCPU, p.x)))) - 1) * p.y)
        }.reduce((x: Array[Double], y: Array[Double]) => daddvv(x, y))
        wCPU = dsubvv(wCPU, gradient)
      }

      (0 until wGPU.length).map(i => {
         assert(abs(wGPU(i) - wCPU(i)) < 1e-7) 
      })

    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run logistic regression with GPU Memory Persistance", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      def dmulvs(x: Array[Double], c: Double) : Array[Double] =
        Array.tabulate(x.length)(i => x(i) * c)
      def daddvv(x: Array[Double], y: Array[Double]) : Array[Double] =
        Array.tabulate(x.length)(i => x(i) + y(i))
      def dsubvv(x: Array[Double], y: Array[Double]) : Array[Double] =
        Array.tabulate(x.length)(i => x(i) - y(i))
      def ddotvv(x: Array[Double], y: Array[Double]) : Double =
        (x zip y).foldLeft(0.0)((a, b) => a + (b._1 * b._2))

      val N = 1024  // Number of data points
      val D = 10   // Numer of dimensions
      val R = 0.7  // Scaling factor
      val ITERATIONS = 5
      val rand = new Random(42)
      val numSlices = 10

      val ptxURL = getClass.getResource("/testCUDAKernels.ptx")
      val mapFunction = sc.broadcast(
        new CUDAFunction(
        "_Z5LRMapPKlPKdS2_PlPdlS2_",
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
        "_Z8LRReducePKlPKdPlPdlii",
        Array("this"),
        Array("this"),
        ptxURL,
        Seq(),
        Some((size: Long) => 1),
        Some(dimensions)))

      def generateData: Array[DataPoint] = {
        def generatePoint(i: Int): DataPoint = {
          val y = if (i % 2 == 0) -1 else 1
          val x = Array.fill(D){rand.nextGaussian + y * R}
          DataPoint(x, y)
        }
        Array.tabulate(N)(generatePoint)
      }

      val points = sc.parallelize(generateData, numSlices)
      val pointsCached = points.cache()

      val w = Array.fill(D){2 * rand.nextDouble - 1}

      var wCPU = Array.tabulate(D)(i => w(i))
      var wGPU = Array.tabulate(D)(i => w(i))
      var wGPUCache = Array.tabulate(D)(i => w(i))

      var startTime = Calendar.getInstance().getTimeInMillis
      for (i <- 1 to ITERATIONS) {
        val gradient = pointsCached.map { p =>
          dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wCPU, p.x)))) - 1) * p.y)
        }.reduce((x: Array[Double], y: Array[Double]) => daddvv(x, y))
        wCPU = dsubvv(wCPU, gradient)
      }
      info("CPU Processing(1) time in milliseconds = " + (Calendar.getInstance().getTimeInMillis - startTime));

      wCPU = Array.tabulate(D)(i => w(i))
      startTime = Calendar.getInstance().getTimeInMillis
      for (i <- 1 to ITERATIONS) {
        val gradient = pointsCached.map { p =>
          dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wCPU, p.x)))) - 1) * p.y)
        }.reduce((x: Array[Double], y: Array[Double]) => daddvv(x, y))
        wCPU = dsubvv(wCPU, gradient)
      }
      info("CPU Processing(2) time in milliseconds = " + (Calendar.getInstance().getTimeInMillis - startTime));


      val pointsColumn = points.convert(ColumnFormat)
      startTime = Calendar.getInstance().getTimeInMillis
      for (i <- 1 to ITERATIONS) {
        val wGPUbc = sc.broadcast(wGPU)
        val gradient = pointsColumn.mapExtFunc((p: DataPoint) =>
          dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wGPUbc.value, p.x)))) - 1) * p.y),
          mapFunction.value, outputArraySizes = Array(D),
          inputFreeVariables = Array(wGPUbc.value)
        ).reduceExtFunc((x: Array[Double], y: Array[Double]) => daddvv(x, y),
                        reduceFunction.value, outputArraySizes = Array(D))
        wGPU = dsubvv(wGPU, gradient)
      }
      info("GPU Processing(1) time in milliseconds = " + (Calendar.getInstance().getTimeInMillis - startTime));


      wGPU = Array.tabulate(D)(i => w(i))
      startTime = Calendar.getInstance().getTimeInMillis
      for (i <- 1 to ITERATIONS) {
        val wGPUbc = sc.broadcast(wGPU)
        val gradient = pointsColumn.mapExtFunc((p: DataPoint) =>
          dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wGPUbc.value, p.x)))) - 1) * p.y),
          mapFunction.value, outputArraySizes = Array(D),
          inputFreeVariables = Array(wGPUbc.value)
        ).reduceExtFunc((x: Array[Double], y: Array[Double]) => daddvv(x, y),
                        reduceFunction.value, outputArraySizes = Array(D))
        wGPU = dsubvv(wGPU, gradient)
      }
      info("GPU Processing(2) time in milliseconds = " + (Calendar.getInstance().getTimeInMillis - startTime));

      points.cacheGpu()
      val pointsColumnCached = points.convert(ColumnFormat).cache()
      startTime = Calendar.getInstance().getTimeInMillis
      for (i <- 1 to ITERATIONS) {
        val wGPUCachebc = sc.broadcast(wGPUCache)
        val gradient = pointsColumnCached.mapExtFunc((p: DataPoint) =>
          dmulvs(p.x,  (1 / (1 + exp(-p.y * (ddotvv(wGPUCachebc.value, p.x)))) - 1) * p.y),
          mapFunction.value, outputArraySizes = Array(D),
          inputFreeVariables = Array(wGPUCachebc.value)
        ).reduceExtFunc((x: Array[Double], y: Array[Double]) => daddvv(x, y),
                        reduceFunction.value, outputArraySizes = Array(D))
        wGPUCache = dsubvv(wGPUCache, gradient)
      }
      info("GPU Cache Processing time in milliseconds = " + (Calendar.getInstance().getTimeInMillis - startTime));
      pointsColumnCached.unCacheGpu()

      (0 until wGPU.length-1).map(i => {
        assert(abs(wGPUCache(i) - wCPU(i)) < 1e-7)
      })
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  // TODO check other formats - cubin and fatbin
  // TODO make the test somehow work on multiple platforms, preferably without recompilation

}
