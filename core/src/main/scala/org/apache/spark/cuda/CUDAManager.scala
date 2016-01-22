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

import scala.collection.mutable.{Map, HashMap, MutableList}

import java.lang.Thread
import java.net.URL
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import jcuda.CudaException;
import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.CUdevice_attribute
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.CUresult
import jcuda.driver.CUstream
import jcuda.driver.JCudaDriver
import jcuda.runtime.{cudaMemcpyKind, cudaStream_t}
import jcuda.runtime.JCuda

import org.apache.commons.io.IOUtils
import org.apache.spark.SparkException
import org.apache.spark.unsafe.memory.Pointer

import org.slf4j.Logger
import org.slf4j.LoggerFactory


object CUDAManagerCachedModule {
  private val cachedModules = new HashMap[(String, Int), CUmodule]
  def getInstance() : HashMap[(String, Int), CUmodule] = { cachedModules }
}


class CUDAManager {
  // Initialization
  // This is supposed to be called before ANY other JCuda* call to ensure we have properly loaded
  // native jCuda library and cuda context
  try {
    JCudaDriver.setExceptionsEnabled(true)
    JCudaDriver.cuInit(0)
  } catch {
    case ex: UnsatisfiedLinkError =>
      throw new SparkException("Could not initialize CUDA, because native jCuda libraries were " +
        "not detected - make sure Driver and Executors are able to load them", ex)
    case ex: NoClassDefFoundError =>
      throw new SparkException("Could not initialize CUDA, because native jCuda libraries were " +
        "not detected - make sure Driver and Executors are able to load them", ex)

    case ex: Throwable =>
      throw new SparkException("Could not initialize CUDA because of unknown reason", ex)
  }

  val deviceCount = {
    // TODO check only those devices with compute capability 2.0+ for streams support and save them
    // in an array
    val cnt = new Array[Int](1)
    JCudaDriver.cuDeviceGetCount(cnt)
    cnt(0)
  }

  private val allStreams = MutableList[Array[cudaStream_t]]()

  private def allocateThreadStreams: Array[cudaStream_t] = {
    val threadStreams = (0 to deviceCount - 1).map { devIx =>
      JCuda.cudaSetDevice(devIx)
      val stream = new cudaStream_t
      JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
      stream
    }.toArray
    synchronized {
      allStreams += threadStreams
    }
    threadStreams
  }

  private val streams: ThreadLocal[Array[cudaStream_t]] = new ThreadLocal[Array[cudaStream_t]] {
    override def initialValue(): Array[cudaStream_t] = {
      if (deviceCount > 0) {
        allocateThreadStreams
      } else {
        new Array[cudaStream_t](0)
      }
    }
  }

  private def getStream(devIx: Int): cudaStream_t = {
    return streams.get.apply(devIx)
  }

  /**
    * Chooses a device to work on and returns a stream for it.
    */
  // TODO make sure only specified amount of tasks at once uses GPU
  // TODO make sure that amount of conversions is minimized by giving GPU to appropriate tasks,
  // task context might be required for that
  private[spark] def getDevice(memoryUsage: Long, gpuDevIx: Int): Int = {
    if (deviceCount == 0) {
      throw new SparkException("No available CUDA devices to create a stream")
    }

    // ensuring streams are already created, since their creation calls cudaSetDevice
    streams.get

    // TODO balance the load (ideally equal amount of streams everywhere but without
    // synchronization) better than just picking at random - this will have standard deviation
    // around sqrt(num_of_threads)
    // maybe correct synchronized load balancing is okay after all - partitions synchronize to
    // allocate the memory anyway
    var startDev = gpuDevIx
    var endDev = gpuDevIx
    if (gpuDevIx < 0) {
      startDev = Thread.currentThread().getId().toInt % deviceCount
      endDev = startDev + deviceCount - 1
    }
    (startDev to endDev).map(_ % deviceCount).map { devIx =>
      JCuda.cudaSetDevice(devIx)
      val memInfo = Array.fill(2)(new Array[Long](1))
      JCuda.cudaMemGetInfo(memInfo(0), memInfo(1))
      val freeMem = memInfo(0)(0)
      if (freeMem >= memoryUsage) {
        if (CUDAManager.logger.isDebugEnabled()) {
          CUDAManager.logger.debug(s"Choosing stream from device $devIx for running the " +
            s"kernel (Thread ID ${Thread.currentThread.getId})");
        }
        // TODO ensure that there will be enough memory available at allocation time (maybe by
        // allocating it now?)
        // TODO GPU memory pooling - no need to reallocate, since usually exact same sizes of memory
        // chunks will be required
        return devIx
      }
    }

    throw new SparkException("No available CUDA devices with enough free memory " +
      s"($memoryUsage bytes needed)")
  }

  private def cachedLoadModule(resource: Either[URL, (String, String)]): CUmodule = {
    var resourceURL: URL = null
    var key: String = null
    var ptxString: String = null
    resource match {
      case Left(resURL) =>
        key = resURL.toString()
        resourceURL = resURL
      case Right((k, v)) => {
        key = k
        ptxString = v
      }
    }

    val devIx = new Array[Int](1)
    JCuda.cudaGetDevice(devIx)
    synchronized {
      // Since multiple modules cannot be loaded into one context in runtime API,
      //   we use singlton cache http://stackoverflow.com/questions/32502375/
      //   loading-multiple-modules-in-jcuda-is-not-working
      // TODO support loading multple ptxs
      //   http://stackoverflow.com/questions/32535828/jit-in-jcuda-loading-multiple-ptx-modules
      CUDAManagerCachedModule.getInstance.getOrElseUpdate((key, devIx(0)), {
        // TODO maybe unload the module if it won't be needed later
        var moduleBinaryData: Array[Byte] = null
        if (resourceURL != null) {
          val inputStream = resourceURL.openStream()
          moduleBinaryData = IOUtils.toByteArray(inputStream)
          inputStream.close()
        } else {
          moduleBinaryData = ptxString.getBytes()
        }

        val moduleBinaryData0 = new Array[Byte](moduleBinaryData.length + 1)
        System.arraycopy(moduleBinaryData, 0, moduleBinaryData0, 0, moduleBinaryData.length)
        moduleBinaryData0(moduleBinaryData.length) = 0
        val module = new CUmodule
        JCudaDriver.cuModuleLoadData(module, moduleBinaryData0)
        module
      })
    }
  }

  private[spark] def allocGPUMemory(size: Long): Pointer = {
    require(size >= 0)
    val ptr = new jcuda.Pointer
    if (CUDAManager.logger.isDebugEnabled()) {
      CUDAManager.logger.debug(s"Allocating ${size}B of GPU memory (Thread ID " +
        s"${Thread.currentThread.getId})");
    }
    val result = JCuda.cudaMalloc(ptr, size)
    if (result != CUresult.CUDA_SUCCESS) {
      throw new CudaException("Cannot allocate GPU memory: " + JCuda.cudaGetErrorString(result));
    }
    assert(size == 0 || ptr != new jcuda.Pointer())
    new Pointer(ptr)
  }

  private[spark] def freeGPUMemory(ptr: Pointer) {
    JCuda.cudaFree(ptr.getJPointer())
  }

  private[spark] def memcpyH2DASync(gpuPtr: Pointer, cpuPtr: Pointer, length: Long, devIx: Int) {
    JCuda.cudaMemcpyAsync(gpuPtr.getJPointer(), cpuPtr.getJPointer(), length,
      cudaMemcpyKind.cudaMemcpyHostToDevice, getStream(devIx))
  }

  private[spark] def memcpyD2HASync(cpuPtr: Pointer, gpuPtr: Pointer, length: Long, devIx: Int) {
    JCuda.cudaMemcpyAsync(cpuPtr.getJPointer(), gpuPtr.getJPointer(), length,
      cudaMemcpyKind.cudaMemcpyDeviceToHost, getStream(devIx))
  }

  private[spark] def memsetASync(gpuPtr: Pointer, value: Byte, length: Long, devIx: Int) {
    JCuda.cudaMemsetAsync(gpuPtr.getJPointer(), value, length, getStream(devIx))
  }

  private[spark] def streamSynchronize(devIx: Int) {
    JCuda.cudaStreamSynchronize(getStream(devIx))
  }

  private[spark] def moduleGetFunction(resource: Any, kernelSignature: String): CUfunction = {
    val module = resource match {
      case url: URL => cachedLoadModule(Left(url))
      case (name: String, ptx: String) => cachedLoadModule(Right(name, ptx))
      case _ => throw new SparkException("Unsupported resource type for CUDAFunction")
    }
    val function = new CUfunction
    JCudaDriver.cuModuleGetFunction(function, module, kernelSignature)
    function
  }

  private[spark] def launchKernel(f: CUfunction,
                                   gridDimX: Int, gridDimY: Int, gridDimZ: Int,
                                   blockDimX: Int, blockDimY: Int, blockDimZ: Int,
                                   sharedMemBytes: Int, devIx: Int,
                                   kernelParams: Pointer) {
    val stream = getStream(devIx)
    val wrappedStream = new CUstream(stream)
    JCudaDriver.cuLaunchKernel(f,
      gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, wrappedStream,
      kernelParams.getJPointer(), null)
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

  /**
   * Release resources connected to CUDA. After this call, this object should not be used again.
   */
  private[spark] def stop() {
    allStreams.flatten.foreach(JCuda.cudaStreamDestroy(_))
  }
}

object CUDAManager {
  private final val logger: Logger = LoggerFactory.getLogger(classOf[CUDAManager])
}
