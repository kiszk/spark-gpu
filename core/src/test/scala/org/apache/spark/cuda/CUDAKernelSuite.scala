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

  private val conf = new SparkConf(false)

  test("Ensure kernel is serializable", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    val resource = getClass.getClassLoader.getResourceAsStream("testCUDAKernels.ptx")
    val ptxData = IOUtils.toByteArray(resource)
    val kernel = new CUDAKernel(
      "_Z8identityPKiPil",
      Array("this"),
      Array("this"),
      ptxData)
    SparkEnv.get.closureSerializer.newInstance().serialize(kernel)
  }

  test("Run identity kernel on a single primitive column", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val resource = getClass.getClassLoader.getResourceAsStream("testCUDAKernels.ptx")
      val ptxData = IOUtils.toByteArray(resource)
      val kernel = new CUDAKernel(
        "_Z8identityPKiPil",
        Array("this"),
        Array("this"),
        ptxData)
      val input = ColumnPartitionDataBuilder.build(1 to 1024)
      val output = kernel.run[Int, Int](input)
      assert(output.size == 1024)
      assert(output.schema.isPrimitive)
      assert(output.iterator.toIndexedSeq.sameElements(1 to 1024))
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
      val resource = getClass.getClassLoader.getResourceAsStream("testCUDAKernels.ptx")
      val ptxData = IOUtils.toByteArray(resource)
      val kernel = new CUDAKernel(
        "_Z12vectorLengthPKdS0_Pdl",
        Array("this.x", "this.y"),
        Array("this.len"),
        ptxData)
      val inputVals = (1 to 100).flatMap { x =>
        (1 to 100).map { y =>
          Vector2DDouble(x, y)
        }
      }
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = kernel.run[Vector2DDouble, VectorLength](input)
      assert(output.size == 10000)
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
      val resource = getClass.getClassLoader.getResourceAsStream("testCUDAKernels.ptx")
      val ptxData = IOUtils.toByteArray(resource)
      val kernel = new CUDAKernel(
        "_Z9plusMinusPKdPKfPdPfl",
        Array("this.base", "this.deviation"),
        Array("this.a", "this.b"),
        ptxData)
      val inputVals = (1 to 100).flatMap { base =>
        (1 to 100).map { deviation =>
          PlusMinus(base * 0.1, deviation * 0.01f)
        }
      }
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = kernel.run[PlusMinus, FloatRange](input)
      assert(output.size == 10000)
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
      val resource = getClass.getClassLoader.getResourceAsStream("testCUDAKernels.ptx")
      val ptxData = IOUtils.toByteArray(resource)
      val kernel = new CUDAKernel(
        "_Z19applyLinearFunctionPKsPslss",
        Array("this"),
        Array("this"),
        ptxData,
        List(2: Short, 3: Short))
      val input = ColumnPartitionDataBuilder.build((1 to 1000).map(_.toShort))
      val output = kernel.run[Short, Short](input)
      assert(output.size == 1000)
      output.iterator.toIndexedSeq.zip((1 to 1000).map(x => (2 + 3 * x).toShort)).foreach {
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
      val resource = getClass.getClassLoader.getResourceAsStream("testCUDAKernels.ptx")
      val ptxData = IOUtils.toByteArray(resource)
      // we only use size/8 GPU threads and run block on a single warp
      val kernel = new CUDAKernel(
        "_Z8blockXORPKcPcll",
        Array("this"),
        Array("this"),
        ptxData,
        List(0x0102030411121314l),
        Some(size => (((size + 32 * 8 - 1) / (32 * 8)).toInt, 32)))
      val inputVals = List.fill(10)(List(
          0x14, 0x13, 0x12, 0x11, 0x04, 0x03, 0x02, 0x01,
          0x34, 0x33, 0x32, 0x31, 0x00, 0x00, 0x00, 0x00
        ).map(_.toByte)).flatten
      val input = ColumnPartitionDataBuilder.build(inputVals)
      val output = kernel.run[Byte, Byte](input)
      assert(output.size == 160)
      val expectedOutputVals = List.fill(10)(List(
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

  // TODO check other formats - cubin and fatbin
  // TODO make the test somehow work on multiple platforms, preferably without recompilation

}
