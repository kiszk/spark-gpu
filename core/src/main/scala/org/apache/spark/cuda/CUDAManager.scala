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
import scala.util.Random

import java.nio.file.{Files, Paths}

import jcuda.Pointer
import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.CUdevice_attribute
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver
import jcuda.runtime.cudaStream_t
import jcuda.runtime.JCuda

import org.apache.commons.io.IOUtils

import org.apache.spark.SparkException

class CUDAManager {

  private val registeredKernels: Map[String, CUDAKernel] = new HashMap[String, CUDAKernel]()

  // Initialization
  JCudaDriver.cuInit(0)

  JCudaDriver.setExceptionsEnabled(true)

  val deviceCount = {
    // TODO check only those devices with compute capability 2.0+ for streams support and save them
    // in an array
    val cnt = new Array[Int](1)
    JCudaDriver.cuDeviceGetCount(cnt)
    cnt(0)
  }

  private val context: Option[CUcontext] = if (deviceCount > 0) {
    val dev = new CUdevice
    JCudaDriver.cuDeviceGet(dev, 0)
    val ctx = new CUcontext
    JCudaDriver.cuCtxCreate(ctx, 0, dev)
    Some(ctx)
  } else {
    None
  }

  private val streams: ThreadLocal[Array[cudaStream_t]] = new ThreadLocal[Array[cudaStream_t]] {
    override def initialValue(): Array[cudaStream_t] = {
      context match {
        case Some(ctx) =>
          (1 to deviceCount).map { devIx =>
            JCuda.cudaSetDevice(devIx)
            val stream = new cudaStream_t
            JCuda.cudaStreamCreate(stream)
            stream
          } .toArray
        case None => new Array[cudaStream_t](0)
      }
    }
  }

  /**
   * Chooses a device to work on and returns a stream for it.
   */
  // TODO make sure only specified amount of tasks at once uses GPU
  // TODO make sure that amount of conversions is minimized by giving GPU to appropriate tasks,
  // task context might be required for that
  private[spark] def getStream(memoryUsage: Long): cudaStream_t = {
    if (deviceCount == 0) {
      throw new SparkException("No available CUDA devices to create a stream")
    }

    // TODO balance the load (ideally equal amount of streams everywhere but without
    // synchronization) better than just picking at random - this will have standard deviation
    // around sqrt(num_of_threads)
    val startDev = Random.nextInt(deviceCount)
    (startDev to (startDev + deviceCount - 1)).map(_ % deviceCount).map { devIx =>
      JCuda.cudaSetDevice(devIx)
      val memInfo = Array.fill(2)(new Array[Long](1))
      JCuda.cudaMemGetInfo(memInfo(0), memInfo(1))
      val freeMem = memInfo(0)(0)
      if (freeMem >= memoryUsage) {
        // TODO ensure that there will be enough memory available at allocation time (maybe by
        // allocating it now?)
        // TODO GPU memory pooling - no need to reallocate, since usually exact same sizes of memory
        // chunks will be required
        return streams.get.apply(devIx)
      }
    }

    throw new SparkException("No available CUDA devices with enough free memory " +
      s"($memoryUsage bytes needed)")
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
      constArgs: Seq[AnyVal] = Seq(),
      dimensions: Option[Long => (Int, Int)] = None): CUDAKernel = {
    val moduleBinaryData = Files.readAllBytes(Paths.get(moduleFilePath))
    registerCUDAKernel(name, kernelSignature, inputColumnsOrder, outputColumnsOrder,
      moduleBinaryData, constArgs, dimensions)
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
      constArgs: Seq[AnyVal] = Seq(),
      dimensions: Option[Long => (Int, Int)] = None): CUDAKernel = {
    val resource = getClass.getClassLoader.getResourceAsStream(resourcePath)
    val moduleBinaryData = IOUtils.toByteArray(resource)
    registerCUDAKernel(name, kernelSignature, inputColumnsOrder, outputColumnsOrder,
      moduleBinaryData, constArgs, dimensions)
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
      constArgs: Seq[AnyVal] = Seq(),
      dimensions: Option[Long => (Int, Int)] = None): CUDAKernel = {
    val kernel = new CUDAKernel(kernelSignature, inputColumnsOrder, outputColumnsOrder,
      moduleBinaryData, constArgs, dimensions)
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
   * Gets the kernel registered with given name. Must not be called when `registerCUDAKernel` might
   * be called at the same time from another thread.
   */
  def getKernel(name: String): CUDAKernel = {
    registeredKernels.applyOrElse(name,
      (n: String) => throw new SparkException(s"Kernel with name $n was not registered"))
  }

  private val cachedModules = new ThreadLocal[HashMap[Array[Byte], CUmodule]] {
    override def initialValue() = new HashMap[Array[Byte], CUmodule]
  }

  private[spark] def cachedLoadModule(moduleBinaryData: Array[Byte]): CUmodule = {
    cachedModules.get.getOrElseUpdate(moduleBinaryData, {
      // TODO maybe unload the module if it won't be needed later
      val module = new CUmodule
      JCudaDriver.cuModuleLoadData(module, moduleBinaryData)
      module
    })
  }

  private[spark] def allocateGPUMemory(size: Long): Pointer = {
    val ptr = new Pointer
    JCuda.cudaMalloc(ptr, size)
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
