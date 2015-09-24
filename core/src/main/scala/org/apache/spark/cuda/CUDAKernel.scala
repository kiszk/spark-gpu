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

import scala.reflect.ClassTag

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.CUstream
import jcuda.driver.JCudaDriver
import jcuda.runtime.cudaStream_t
import jcuda.runtime.cudaMemcpyKind
import jcuda.runtime.JCuda

import org.apache.spark.{PartitionData, ColumnPartitionData, ColumnPartitionSchema, SparkEnv}
import org.apache.spark.util.Utils

/**
 * A CUDA kernel wrapper. Contains CUDA module, information how to extract CUDA kernel from it and
 * how to order input/output columns for the kernel invocation.
 * The kernel should take a long size parameter, then pointers to C arrays with the input data, then
 * pointers to C arrays with output data, e.g.
 * `void identity(long size, const int *input, int *output)`. The kernel should work when the total
 * number of threads is larger than size, since number of threads will have to be aligned at least
 * to multiples of 32.
 *
 * @param kernelSignature The C++-style signature of the kernel function. To figure it out you might
 * want to compile the cuda file with nvcc's `-Xptxas="-v"` option.
 * @param inputColumnsOrder Order in which columns of the input partition should be passed to the
 * kernel, e.g. `Array("this")` for `RDD[Int]` or `Array("this.x", "this.y")` for `RDD[Point]`,
 * where `Point` is `case class Point(x: Int, y: Int)`. The name is string with how you could access
 * given property in the object.
 * @param outputColumnsOrder Order in which columns of the output partition should be passed to the
 * kernel. See inputColumnsOrder for format details.
 * @param moduleBinaryData Binary data of a compiled CUDA module in cubin, PTX or fatbin format.
 * @param dimensions Optional function to compute thread dimensions for running the kernel. By
 * default it is assumed that each thread computes one value with its index (blockSize * blockId +
 * threadId) and dimensions are automatically computed to maximize block size.
 */
// TODO since this is serialized and sent together with an RDD, it might be a good idea to avoid
// sending the same module data with multiple used kernels many times
class CUDAKernel(
    val kernelSignature: String,
    val inputColumnsOrder: Seq[String],
    val outputColumnsOrder: Seq[String],
    val moduleBinaryData: Array[Byte],
    val dimensions: Option[Long => (Int, Int)] = None) {

  var cachedFunction: Option[CUfunction] = None

  private[spark] def run[T: ClassTag, U: ClassTag](in: ColumnPartitionData[T]):
      ColumnPartitionData[U] = {

    val outputSchema = ColumnPartitionSchema.schemaFor[U]

    val memoryUsage = in.memoryUsage + outputSchema.memoryUsage(in.size)

    val context = SparkEnv.get.cudaManager.acquireContext(memoryUsage)

    Utils.tryWithSafeFinally {
      val function = cachedFunction.getOrElse {
        // TODO maybe unload the module if it won't be needed later
        val module = new CUmodule
        JCudaDriver.cuModuleLoadData(module, moduleBinaryData)
        val f = new CUfunction
        JCudaDriver.cuModuleGetFunction(f, module, kernelSignature)
        cachedFunction = Some(f)
        f
      }

      var gpuInputPtrs = Vector[Pointer]()
      var gpuOutputPtrs = Vector[Pointer]()
      Utils.tryWithSafeFinally {
        for (col <- in.schema.orderedColumns(inputColumnsOrder)) {
          System.err.println("Allocaling " + col.memoryUsage(in.size) + " bytes on gpu for input " + col.prettyAccessor)
          gpuInputPtrs = gpuInputPtrs :+
            SparkEnv.get.cudaManager.allocateGPUMemory(col.memoryUsage(in.size))
        }

        for (col <- outputSchema.orderedColumns(outputColumnsOrder)) {
          System.err.println("2Allocaling " + col.memoryUsage(in.size) + " bytes on gpu for output " + col.prettyAccessor)
          gpuOutputPtrs = gpuOutputPtrs :+
            SparkEnv.get.cudaManager.allocateGPUMemory(col.memoryUsage(in.size))
        }

        var out = new ColumnPartitionData[U](outputSchema, in.size)

        val stream = new cudaStream_t
        JCuda.cudaStreamCreate(stream)

        Utils.tryWithSafeFinally {
          System.err.println("in.size " + in.size)

          for ((cpuPtr, gpuPtr, col) <- (in.pointers, gpuInputPtrs, in.schema.columns).zipped) {
            System.err.println("Async cpy " + col.prettyAccessor + " bytes " + col.memoryUsage(in.size))
            JCuda.cudaMemcpyAsync(gpuPtr, cpuPtr, col.memoryUsage(in.size),
              cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
          }

          val gpuPtrParams = (gpuInputPtrs ++ gpuOutputPtrs).map(Pointer.to(_))
          val kernelParameters = Pointer.to((Pointer.to(Array(in.size)) +: gpuPtrParams): _*)

          val (gpuGridSize, gpuBlockSize) = dimensions match {
            case Some(computeDim) => computeDim(in.size)
            case None => SparkEnv.get.cudaManager.computeDimensions(in.size)
          }

          JCudaDriver.cuLaunchKernel(
            function,
            gpuGridSize, 1, 1,
            gpuBlockSize, 1, 1,
            0,
            new CUstream(stream),
            kernelParameters, null)

          for ((cpuPtr, gpuPtr, col) <- (in.pointers, gpuInputPtrs, out.schema.columns).zipped) {
            JCuda.cudaMemcpyAsync(gpuPtr, cpuPtr, col.memoryUsage(in.size),
              cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
          }

          JCuda.cudaStreamSynchronize(stream)

          // TODO in.free?

          out
        } {
          JCuda.cudaStreamDestroy(stream)
        }
      } {
        for (ptr <- gpuInputPtrs) {
          SparkEnv.get.cudaManager.deallocateGPUMemory(ptr)
        }
        for (ptr <- gpuOutputPtrs){
          SparkEnv.get.cudaManager.deallocateGPUMemory(ptr)
        }
      }
    } {
      SparkEnv.get.cudaManager.releaseContext(context)
    }
  }

}
