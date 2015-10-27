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

import scala.math._

case class Vector2DDouble(x: Double, y: Double)
case class VectorLength(len: Double)
case class PlusMinus(base: Double, deviation: Float)
case class FloatRange(a: Double, b: Float)

class CUDAKernelSuite extends SparkFunSuite with LocalSparkContext {

  private val conf = new SparkConf(false).set("spark.driver.maxResultSize", "2g")

  test("Ensure kernel is serializable", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val kernel = new CUDAKernel(
      "_Z8identityPKiPil",
      Array("this"),
      Array("this"),
      "testCUDAKernels.ptx", CUDAKernelPtxResource)
    SparkEnv.get.closureSerializer.newInstance().serialize(kernel)
  }

  test("Run identity kernel on a single primitive column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val kernel = new CUDAKernel(
        "_Z8identityPKiPil",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource)
      val n = 1024
      val input = ColumnPartitionDataBuilder.build(1 to n)
      val output = kernel.run[Int, Int](input)
      assert(output.size == n)
      assert(output.schema.isPrimitive)
      assert(output.iterator.toIndexedSeq.sameElements(1 to n))
      input.free
      output.free
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run vectorLength kernel on 2 col -> 1 col", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val kernel = new CUDAKernel(
        "_Z12vectorLengthPKdS0_Pdl",
        Array("this.x", "this.y"),
        Array("this.len"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource)
      val n = 100
      val inputVals = (1 to n).flatMap { x =>
        (1 to n).map { y =>
          Vector2DDouble(x, y)
        }
      }
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = kernel.run[Vector2DDouble, VectorLength](input)
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

  test("Run plusMinus kernel on 2 col -> 2 col", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val kernel = new CUDAKernel(
        "_Z9plusMinusPKdPKfPdPfl",
        Array("this.base", "this.deviation"),
        Array("this.a", "this.b"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource)
      val n = 100
      val inputVals = (1 to n).flatMap { base =>
        (1 to n).map { deviation =>
          PlusMinus(base * 0.1, deviation * 0.01f)
        }
      }
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = kernel.run[PlusMinus, FloatRange](input)
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

  test("Run applyLinearFunction kernel on 1 col + 2 const arg -> 1 col", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val kernel = new CUDAKernel(
        "_Z19applyLinearFunctionPKsPslss",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        List(2: Short, 3: Short))
      val n = 1000
      val input = ColumnPartitionDataBuilder.build((1 to n).map(_.toShort))
      val output = kernel.run[Short, Short](input)
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

  test("Run blockXOR kernel on 1 col + 1 const arg -> 1 col on custom dimensions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      // we only use size/8 GPU threads and run block on a single warp
      val kernel = new CUDAKernel(
        "_Z8blockXORPKcPcll",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        List(0x0102030411121314l),
        None,
        Some((size: Long, stage: Int) => (((size + 32 * 8 - 1) / (32 * 8)).toInt, 32)))
      val n = 10
      val inputVals = List.fill(n)(List(
          0x14, 0x13, 0x12, 0x11, 0x04, 0x03, 0x02, 0x01,
          0x34, 0x33, 0x32, 0x31, 0x00, 0x00, 0x00, 0x00
        ).map(_.toByte)).flatten
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = kernel.run[Byte, Byte](input)
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

  test("Run sum kernel on 1 col -> 1 col in 2 stages", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val kernel = new CUDAKernel(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))
      val n = 30000
      val input = ColumnPartitionDataBuilder.build(1 to n)
      val output = kernel.run[Int, Int](input, Some(1))
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
      val mapKernel = new CUDAKernel(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource)

      val n = 10
      val output = sc.parallelize(1 to n, 1)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
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
      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceKernel = new CUDAKernel(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 10
      val output = sc.parallelize(1 to n, 1)
        .convert(ColumnFormat)
        .reduceUsingKernel((x: Int, y: Int) => x + y, reduceKernel)
      assert(output == n * (n + 1) / 2)
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + reduce on rdds - single partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val mapKernel = new CUDAKernel(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx")

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceKernel = new CUDAKernel(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 10
      val output = sc.parallelize(1 to n, 1)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .reduceUsingKernel((x: Int, y: Int) => x + y, reduceKernel)
      assert(output == n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map on rdds with 100,000,000 elements - multiple partition", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val mapKernel = new CUDAKernel(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx")

      val n = 100000000
      val output = sc.parallelize(1 to n, 64)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
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
      val mapKernel = new CUDAKernel(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx")

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceKernel = new CUDAKernel(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100
      val output = sc.parallelize(1 to n, 16)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .reduceUsingKernel((x: Int, y: Int) => x + y, reduceKernel)
      assert(output == n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + reduce on rdds with 100,000,000 elements - multiple partitions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val mapKernel = new CUDAKernel(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx")

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceKernel = new CUDAKernel(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100000000
      val output = sc.parallelize(1 to n, 64)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .reduceUsingKernel((x: Int, y: Int) => x + y, reduceKernel)
      assert(output == n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + map + reduce on rdds - multiple partitions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val mapKernel = new CUDAKernel(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx")

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceKernel = new CUDAKernel(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100
      val output = sc.parallelize(1 to n, 16)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .reduceUsingKernel((x: Int, y: Int) => x + y, reduceKernel)
      assert(output == 2 * n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  test("Run map + map + map + reduce on rdds - multiple partitions", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val mapKernel = new CUDAKernel(
        "_Z11multiplyBy2PiS_l",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx")

      val dimensions = (size: Long, stage: Int) => stage match {
        case 0 => (64, 256)
        case 1 => (1, 1)
      }
      val reduceKernel = new CUDAKernel(
        "_Z3sumPiS_lii",
        Array("this"),
        Array("this"),
        "testCUDAKernels.ptx", CUDAKernelPtxResource,
        Seq(),
        Some((size: Long) => 2),
        Some(dimensions))

      val n = 100
      val output = sc.parallelize(1 to n, 16)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .mapUsingKernel((x: Int) => 2 * x, mapKernel)
        .reduceUsingKernel((x: Int, y: Int) => x + y, reduceKernel)
      assert(output == 4 * n * (n + 1))
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

  // TODO check other formats - cubin and fatbin
  // TODO make the test somehow work on multiple platforms, preferably without recompilation

}
