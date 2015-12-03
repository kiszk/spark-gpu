package org.apache.spark.cuda

import jcuda.Pointer

import scala.collection.mutable.HashMap

/**
 * Created by kmadhu on 2/12/15.
 */
class GPUMemoryManager(val isDriver : Boolean) {

  val cachedGPUPointers = new HashMap[String, Pointer]()

  def getCachedGPUPointers = cachedGPUPointers

}
