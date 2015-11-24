package org.apache.spark.examples
//package org.madhu

import java.io.Serializable
import java.nio.file.{Paths, Files}

import jcuda.runtime.{JCuda, cudaStream_t}
import jcuda.Pointer
import org.apache.spark.{SparkEnv, SparkConf, SparkContext}
import org.apache.spark.cuda._

import jcuda.driver._
import jcuda.driver.JCudaDriver._

import scala.reflect.ClassTag


/**
 * Created by kmadhu on 26/10/15.
 */
object SparkGPUTest {

  abstract class CUDAKernel1[U : ClassTag,T : ClassTag](moduleFilePath : String, fname : String)
    extends CUDAKernel[U,T](moduleFilePath,fname) {

    val moduleBinaryData = Files.readAllBytes(Paths.get(moduleFilePath))

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

    def run(hyIter : hybridIterator[T], function: CUfunction,stream : cudaStream_t, cuStream : CUstream) : Iterator[U]
  }


  class myCUDAKernel(modulePath : String, fname : String) extends CUDAKernel1[Int,Int](modulePath,fname) {

    override def run(hyIter : hybridIterator[Int], function: CUfunction,stream : cudaStream_t, cuStream : CUstream) : Iterator[Int] = {

     // set GPU Parameters
      val (numElements_GPU)    = cloneFromCPUMemory[Int](Array.fill(1)(hyIter.numElements),4,cuStream)
      val (inputArray_GPU,sz)  = hyIter.getColumnGPUMemory[Int](hyIter, (x=>x), cuStream)
      val results_GPU          = allocateGPUMemory(sz)
      val kernelParameters     = Pointer.to( Pointer.to(numElements_GPU), Pointer.to(inputArray_GPU), Pointer.to(results_GPU) );

      // TODO make launchKernel to accept scala arrays, which would avoid above step
      launchKernel(function , hyIter.numElements, kernelParameters,cuStream)


      // TODO automate this - do it inside launchKernel
      val results_CPU = cloneFromGPUMemory[Int](results_GPU,hyIter.numElements,sz,cuStream)

      JCuda.cudaStreamSynchronize(stream)
      cuMemFree(numElements_GPU);
      cuMemFree(inputArray_GPU);


      // User has the flexibility build any object from the results for GPU kernel
      println("GPU OUTPUT" + results_CPU.toList)
      new hybridIterator(results_CPU.map(_.asInstanceOf[Int]))

    }

  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setMaster("local").setAppName("runJob")
    val sc = new SparkContext(conf)

    val baseRDD = sc.parallelize(1 to 10)
    val k = new myCUDAKernel("/home/kmadhu/jcuda/vecAdd/src/main/resources/kernel.ptx", "square")

    // TODO sc.cudaManager.registerCUDAKernel("square",k)

    val mapRDD =  baseRDD.mapGPU(x => x + x, k).mapGPU(x=>x+x, k)

    println("RESULT="+ mapRDD.collect().toList)

  }
}

