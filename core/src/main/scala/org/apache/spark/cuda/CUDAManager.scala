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

import scala.collection.mutable.{Map, HashMap}

import java.nio.file.{Files, Paths}

import jcuda.Pointer
import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.CUdevice_attribute
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver
import jcuda.runtime.JCuda

import org.apache.commons.io.IOUtils

import org.apache.spark.SparkException

class CUDAManager {

  private val registeredKernels: Map[String, CUDAKernel] = new HashMap[String, CUDAKernel]()

  // Initialization
  JCudaDriver.cuInit(0)

  JCudaDriver.setExceptionsEnabled(true)

  val deviceCount = {
    // TODO check only those devices with compute capability 2.0+ for streams support
    val cnt = new Array[Int](1)
    JCudaDriver.cuDeviceGetCount(cnt)
    cnt(0)
  }

  /**
   * The device we'll be using next. To split the work evenly we increment this value each time.
   */
  private var currentDevice = 0

  private[spark] def acquireContext(memoryUsage: Long): CUcontext = {
    if (deviceCount == 0) {
      throw new SparkException("Trying to acquire CUDA context when no CUDA devices are available")
    }

    // TODO pool contexts per device, since threads can share context
    // TODO make sure only specified amount of tasks at once uses GPU
    // TODO make sure that amount of conversions is minimized by giving GPU to appropriate tasks
    val device = synchronized {
      val dev = new CUdevice
      JCudaDriver.cuDeviceGet(dev, currentDevice)
      currentDevice = (currentDevice + 1) % deviceCount
      dev
    }

    val context = new CUcontext
    JCudaDriver.cuCtxCreate(context, 0, device)
    context
  }

  private[spark] def releaseContext(context: CUcontext) {
    JCudaDriver.cuCtxDestroy(context)
  }

  /**
   * Register in the manager a CUDA kernel from a module file in cubin, PTX or fatbin format.
   */
  def registerCUDAKernelFromFile(
      name: String,
      kernelSignature: String,
      inputColumnsOrder: Seq[String],
      outputColumnsOrder: Seq[String],
      moduleFilePath: String,
      dimensions: Option[Long => (Int, Int)] = None): CUDAKernel = {
    val moduleBinaryData = Files.readAllBytes(Paths.get(moduleFilePath))
    registerCUDAKernel(name, kernelSignature, inputColumnsOrder, outputColumnsOrder,
      moduleBinaryData, dimensions)
  }

  /**
   * Register in the manager a CUDA kernel from a module in resources.
   */
  def registerCUDAKernelFromResource(
      name: String,
      kernelSignature: String,
      inputColumnsOrder: Seq[String],
      outputColumnsOrder: Seq[String],
      resourcePath: String,
      dimensions: Option[Long => (Int, Int)] = None): CUDAKernel = {
    val resource = getClass.getClassLoader.getResourceAsStream(resourcePath)
    val moduleBinaryData = IOUtils.toByteArray(resource)
    registerCUDAKernel(name, kernelSignature, inputColumnsOrder, outputColumnsOrder,
      moduleBinaryData, dimensions)
  }

  /**
   * Register in the manager a CUDA kernel from a module in binary cubin, PTX or fatbin binary data
   * format.
   */
  def registerCUDAKernel(
      name: String,
      kernelSignature: String,
      inputColumnsOrder: Seq[String],
      outputColumnsOrder: Seq[String],
      moduleBinaryData: Array[Byte],
      dimensions: Option[Long => (Int, Int)] = None): CUDAKernel = {
    val kernel = new CUDAKernel(kernelSignature, inputColumnsOrder, outputColumnsOrder,
      moduleBinaryData, dimensions)
    registerCUDAKernel(name, kernel)
    kernel
  }

  /**
   * Register given CUDA kernel in the manager.
   */
  def registerCUDAKernel(name: String, kernel: CUDAKernel) {
    if (registeredKernels.contains(name)) {
      throw new SparkException(s"Kernel with name $name already registered")
    }

    synchronized {
      registeredKernels.put(name, kernel)
    }
  }

  /**
   * Gets the kernel registered with given name
   */
  def getKernel(name: String): CUDAKernel = {
    synchronized {
      registeredKernels.applyOrElse(name,
        (n: String) => throw new SparkException(s"Kernel with name $n was not registered"))
    }
  }

  private[spark] def allocateGPUMemory(size: Long): Pointer = {
    val ptr = new Pointer
    JCuda.cudaMalloc(ptr, size)
    assert(ptr != new Pointer)
    ptr
  }

  private[spark] def deallocateGPUMemory(ptr: Pointer) {
    JCuda.cudaFree(ptr)
  }

  private[spark] def computeDimensions(size: Long): (Int, Int) = {
    val maxBlockDim = {
      val dev = new CUdevice
      JCudaDriver.cuCtxGetDevice(dev)
      val dim = new Array[Int](1)
      JCudaDriver.cuDeviceGetAttribute(dim, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        dev)
      dim(0)
    }
    assert(size <= maxBlockDim * Int.MaxValue.toLong)
    (((size + maxBlockDim - 1) / maxBlockDim).toInt, maxBlockDim)
  }

}
