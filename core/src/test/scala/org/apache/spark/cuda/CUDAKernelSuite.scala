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

class CUDAKernelSuite extends SparkFunSuite with LocalSparkContext {

  private val conf = new SparkConf(false)

  test("Run simple kernel", GPUTest) {
    sc = new SparkContext("local", "test", conf)
    val manager = new CUDAManager
    if (manager.deviceCount > 0) {
      val resource = getClass.getClassLoader.getResourceAsStream("identity.ptx")
      val ptxData = IOUtils.toByteArray(resource)
      val kernel = new CUDAKernel(
        "_Z8identitylPiS_",
        Array("this"),
        Array("this"),
        ptxData)
      val input = ColumnPartitionDataBuilder.build[Int](1 to 1024)
      val output = kernel.run[Int, Int](input)
      assert(output.size == 1024)
      assert(output.schema.isPrimitive)
      assert(output.iterator.toIndexedSeq.sameElements(1 to 1024))
      input.free
      output.free
    }
  }

}
