package org.apache.spark.cuda

import java.nio.file.{Paths, Files}

import jcuda.driver.JCudaDriver._
import java.io._
import jcuda._
import jcuda.driver._
import jcuda.runtime.{cudaStream_t, JCuda}
import org.apache.spark.{SparkException, SparkEnv}

import scala.collection.mutable.{HashMap, Map}
import scala.reflect.ClassTag
import scala.reflect.runtime._

/**
 * Created by kmadhu on 26/10/15.
 */

 abstract class CUDAKernel[U : ClassTag,T : ClassTag](moduleFilePath : String, fname : String)
  extends Serializable {

  /* val moduleBinaryData = Files.readAllBytes(Paths.get(moduleFilePath))

  case class hybridIterator[T: ClassTag](arr: Array[T]) extends Iterator[T] {

    val numElements = arr.length
    var idx: Int = -1

    def hasNext = idx < arr.length - 1

    def next = {
      idx = idx + 1
      arr(idx)
    }

    def getColumnHostMemory[T](hyIter : hybridIterator[T], lamda : (T => Any)) = {
      hyIter.arr.map(lamda)
    }

    def getColumnGPUMemory[T](hyIter : hybridIterator[T], lamda : (T => Any),cuStream : CUstream) = {
      val hostArr = hyIter.arr.map(lamda)
      val (_, hostPtr, sz) = getPointer(hostArr)
      val gpuPtr = allocateGPUMemory(sz)
      cuMemcpyHtoDAsync(gpuPtr, hostPtr, sz,cuStream)
      (gpuPtr, sz)
    }
  }


  def allocateGPUMemory(sz : Int) = {
    val deviceInput = new CUdeviceptr();
    cuMemAlloc(deviceInput, sz)
    deviceInput
  }

  def getPointer(a : Array[Any]) = {
    val (hptr, ptr, sz) = a(0) match {
      case h if h.isInstanceOf[Byte]    => { val hptr = a.map(_.asInstanceOf[Byte]);   ( hptr, Pointer.to(hptr) , a.length * 1) }
      case h if h.isInstanceOf[Short]   => { val hptr = a.map(_.asInstanceOf[Short]);  ( hptr, Pointer.to(hptr) , a.length * 2) }
      case h if h.isInstanceOf[Int]     => { val hptr = a.map(_.asInstanceOf[Int]);    ( hptr, Pointer.to(hptr) , a.length * 4) }
      case h if h.isInstanceOf[Float]   => { val hptr = a.map(_.asInstanceOf[Float]);  ( hptr, Pointer.to(hptr) , a.length * 4) }
      case h if h.isInstanceOf[Long]    => { val hptr = a.map(_.asInstanceOf[Long]);   ( hptr, Pointer.to(hptr) , a.length * 8) }
      case h if h.isInstanceOf[Double]  => { val hptr = a.map(_.asInstanceOf[Double]); ( hptr, Pointer.to(hptr) , a.length * 8) }
    }
    (hptr, ptr,sz)
  }

  def cloneFromGPUMemory[V : ClassTag](deviceOutput : CUdeviceptr,numElements : Int, sz : Int, cuStream : CUstream) = {
    // Allocate host output memory and copy the device output to the host.
    val (hostOutput,ptr,_)  = getPointer( new Array[V](numElements).map(_.asInstanceOf[Any]))
    cuMemcpyDtoHAsync(ptr,deviceOutput,sz,cuStream)
    hostOutput
  }

  def cloneFromCPUMemory[V : ClassTag](hostMem : Array[V],sz : Int, cuStream : CUstream) = {
    val deviceMem = allocateGPUMemory(sz)
    cuMemcpyHtoDAsync(deviceMem,getPointer(hostMem.map(_.asInstanceOf[Any]))._2,sz,cuStream)
    deviceMem
  }

  // asynchronous Launch of kernel
  def launchKernel(function : CUfunction,numElements : Int, kernelParameters : Pointer, cuStream : CUstream) = {
    cuLaunchKernel(function,
      Math.ceil(numElements / 32.0).toInt, 1, 1, // Grid dimension
      32, 1, 1, // Block dimension
      0, cuStream, // Shared memory size and stream
      kernelParameters, null // Kernel- and extra parameters
    );
  }

  def compute(inp: Iterator[T]) : Iterator[U] = {
    val function = SparkEnv.get.cudaManager.getKernelFunction(moduleBinaryData,fname)
    val stream = new cudaStream_t
    JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
    val cuStream = new CUstream(stream)

    val h = inp match {
      case h : hybridIterator[T] => { println("Using the hybrid Iterator from previous RDD") ; h }
      case abc : Iterator[T]  => { println("Converting Regular Iterator to hybridIterator"); new hybridIterator[T](abc.toArray) }
    }
    val result = run(h,function,stream,cuStream)
    JCuda.cudaStreamDestroy(stream)
    result
  } */

    def compute(inp: Iterator[T]) : Iterator[U]
    //def run(inp : hybridIterator[T], function: CUfunction,stream : cudaStream_t, cuStream : CUstream) : Iterator[U]
   // def run(inp: hybridIterator[T],cuFunction: CUfunction): Iterator[U]
 }
