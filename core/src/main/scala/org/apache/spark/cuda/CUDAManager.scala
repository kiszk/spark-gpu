package org.apache.spark.cuda

import java.nio.file.{Paths, Files}

import jcuda.driver.JCudaDriver._
import jcuda.driver._
import jcuda.runtime.JCuda
import org.apache.spark.SparkException

import scala.collection.mutable.{Map, HashMap}

/**
 * Created by kmadhu on 27/10/15.
 */
class CUDAManager (isExecutor : Boolean) {


  private val registeredKernels = new HashMap[String, CUDAKernel[_,_]]()

  /**
   * Register given CUDA kernel in the manager.
   */
  def registerCUDAKernel(name: String, kernel: CUDAKernel[_,_]) {
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
  def getKernel(name: String): CUDAKernel[_,_] = {
    registeredKernels.applyOrElse(name,
      (n: String) => throw new SparkException(s"Kernel with name $n was not registered"))
  }

  /**
   * Load CUDA kernel from a module file in cubin, PTX or fatbin format.
   * To be called from executor side
   */

  if (isExecutor)
  {
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

      case ex: Exception =>
        throw new SparkException("Could not initialize CUDA because of unknown reason", ex)
    }
  }

  private val cachedModules = new ThreadLocal[HashMap[Array[Byte], CUmodule]] {
    override def initialValue() = new HashMap[Array[Byte], CUmodule]
  }

  private[spark] def loadModule(moduleBinaryData: Array[Byte]): CUmodule = {
    cachedModules.get.getOrElseUpdate((moduleBinaryData), {
      // TODO maybe unload the module if it won't be needed later
      // Load the module
      // get Device
      val device = new CUdevice(); cuDeviceGet(device, 0);

      // create a context on Device
      val context = new CUcontext(); cuCtxCreate(context, 0, device);

      val module = new CUmodule;
      JCudaDriver.cuModuleLoadData(module, moduleBinaryData)
      module
    })
  }

  def getKernelFunction(moduleBinaryData : Array[Byte], fname : String) = {
    // Load module
    val module = loadModule(moduleBinaryData)
    // Obtain a function pointer to the "add" function.
    val function = new CUfunction();
    cuModuleGetFunction(function, module, fname);
    function
  }

  def stop() = {

  }



}
