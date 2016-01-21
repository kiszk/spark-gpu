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

import java.nio.ByteBuffer

import org.apache.spark._
import org.apache.spark.util.Utils
import org.apache.spark.unsafe.memory.Pointer

class CUDAManagerSuite extends SparkFunSuite with LocalSparkContext {

  private val conf = new SparkConf(false)

  test("Allocate and copy memory to/from gpu", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = SparkEnv.get.cudaManager
    if (manager.deviceCount > 0) {
      val devIx = manager.getDevice(1024, -1)

      val gpuPtr = manager.allocGPUMemory(1024)
      Utils.tryWithSafeFinally {
        manager.memcpyH2DASync(gpuPtr, Pointer.to(Array.fill[Byte](1024)(42)), 1024, devIx)
        val arr = new Array[Byte](1024)
        manager.memcpyD2HASync(Pointer.to(ByteBuffer.wrap(arr)), gpuPtr, 1024, devIx)
        manager.streamSynchronize(devIx)
        assert(arr.forall(_ == 42))
      } {
        manager.freeGPUMemory(gpuPtr)
      }
    } else {
      info("No CUDA devices, so skipping the test.")
    }
  }

}
