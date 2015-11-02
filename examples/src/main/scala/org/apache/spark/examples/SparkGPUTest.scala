package org.apache.spark.examples
//package org.madhu

import jcuda.runtime.{JCuda, cudaStream_t}
import jcuda.Pointer
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.cuda._

import jcuda.driver._
import jcuda.driver.JCudaDriver._


/**
 * Created by kmadhu on 26/10/15.
 */
object SparkGPUTest {

  class myCUDAKernel(modulePath : String, fname : String) extends CUDAKernel[Int,Int](modulePath,fname) {

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

