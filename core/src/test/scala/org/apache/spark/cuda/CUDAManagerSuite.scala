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

import org.apache.spark._

class CUDAManagerSuite extends SparkFunSuite with LocalSparkContext {

  private val conf = new SparkConf(false)

  test("Registering a kernel", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = SparkEnv.get.cudaManager
    if (manager.deviceCount > 0) {
      val kernel = manager.registerCUDAKernelFromResource(
        "identity",
        "_Z8identitylPiS_",
        Array("this"),
        Array("this"),
        "identity.ptx")
      assert(manager.getKernel("identity") == kernel)
    }
  }

  test("Allocate and copy memory to/from gpu", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = SparkEnv.get.cudaManager
    if (manager.deviceCount > 0) {
      val context = manager.acquireContext(1024)

      val gpuPtr = manager.allocateGPUMemory(1024)
      JCuda.cudaMemcpy(gpuPtr, Pointer.to(Array.fill[Byte](1024)(42)), 1024,
        cudaMemcpyKind.cudaMemcpyHostToDevice)
      val arr = new Array[Byte](1024)
      JCuda.cudaMemcpy(Pointer.to(ByteBuffer.wrap(arr)), gpuPtr, 1024,
        cudaMemcpyKind.cudaMemcpyHostToDevice)

      assert(arr.forall(_ == 42))

      manager.releaseContext(context)
    }
  }

}
