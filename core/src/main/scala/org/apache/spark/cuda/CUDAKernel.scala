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

import org.apache.spark.{PartitionData, ColumnPartitionData, ColumnPartitionSchema, SparkEnv,
  SparkException}
import org.apache.spark.util.Utils

/**
 * A CUDA kernel wrapper. Contains CUDA module, information how to extract CUDA kernel from it and
 * how to order input/output columns for the kernel invocation.
 * The kernel should take pointers to C arrays with the input data, then long size parameter, then
 * optionally all constant parameters of respective types, e.g.
 * `void identity(const int *input, int *output, long size, short unusedParam)`. The kernel should
 * work when the total number of threads is larger than size, since number of threads will have to
 * be aligned at least to multiples of 32.
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
 * @param constArgs Optional list of constant arguments supplied to the kernel.
 * @param dimensions Optional function to compute thread dimensions for running the kernel. By
 * default it is assumed that each thread computes one value with its index (blockSize * blockId +
 * threadId) and dimensions are automatically computed to maximize block size.
 */
// TODO allow kernel to use some of input memory in-place, i.e. reuse input as output and put those
// buffers later to output ColumnPartitionData - can do it by special
// inputColumnOrder/outputColumnsOrder syntax
// TODO improve the way constant arguments are passed - especially duplication of kernels
class CUDAKernel(
    val kernelSignature: String,
    val inputColumnsOrder: Seq[String],
    val outputColumnsOrder: Seq[String],
    val moduleBinaryData: Array[Byte],
    val constArgs: Seq[AnyVal] = Seq(),
    val dimensions: Option[Long => (Int, Int)] = None) {

  var cachedFunction: Option[CUfunction] = None

  private[spark] def run[T: ClassTag, U: ClassTag](in: ColumnPartitionData[T]):
      ColumnPartitionData[U] = {

    val outputSchema = ColumnPartitionSchema.schemaFor[U]

    val memoryUsage = in.memoryUsage + outputSchema.memoryUsage(in.size)

    val stream = SparkEnv.get.cudaManager.getStream(memoryUsage)

    val function = cachedFunction.getOrElse {
      val module = SparkEnv.get.cudaManager.cachedLoadModule(moduleBinaryData)
      val f = new CUfunction
      JCudaDriver.cuModuleGetFunction(f, module, kernelSignature)
      cachedFunction = Some(f)
      f
    }

    val out = new ColumnPartitionData[U](outputSchema, in.size)
    try {
      var gpuInputPtrs = Vector[Pointer]()
      var gpuOutputPtrs = Vector[Pointer]()
      Utils.tryWithSafeFinally {
        val inColumns = in.schema.orderedColumns(inputColumnsOrder)
        for (col <- inColumns) {
          gpuInputPtrs = gpuInputPtrs :+
            SparkEnv.get.cudaManager.allocateGPUMemory(col.memoryUsage(in.size))
        }

        val outColumns = out.schema.orderedColumns(outputColumnsOrder)
        for (col <- outColumns) {
          gpuOutputPtrs = gpuOutputPtrs :+
            SparkEnv.get.cudaManager.allocateGPUMemory(col.memoryUsage(in.size))
        }

        val inPointers = in.orderedPointers(inputColumnsOrder)
        for ((cpuPtr, gpuPtr, col) <- (inPointers, gpuInputPtrs, inColumns).zipped) {
          JCuda.cudaMemcpyAsync(gpuPtr, cpuPtr, col.memoryUsage(in.size),
            cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        }

        val gpuPtrParams = (gpuInputPtrs ++ gpuOutputPtrs).map(Pointer.to(_))
        val sizeParam = List(Pointer.to(Array(in.size)))
        val constArgParams = constArgs.map {
          case v: Byte => Pointer.to(Array(v))
          case v: Char => Pointer.to(Array(v))
          case v: Short => Pointer.to(Array(v))
          case v: Int => Pointer.to(Array(v))
          case v: Long => Pointer.to(Array(v))
          case v: Float => Pointer.to(Array(v))
          case v: Double => Pointer.to(Array(v))
          case _ => throw new SparkException("Unsupported type passed to kernel as a constant "
            + "argument")
        }
        val params = gpuPtrParams ++ sizeParam ++ constArgParams
        val kernelParameters = Pointer.to(params: _*)

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

        val outPointers = out.orderedPointers(outputColumnsOrder)
        for ((cpuPtr, gpuPtr, col) <- (outPointers, gpuOutputPtrs, outColumns).zipped) {
          JCuda.cudaMemcpyAsync(cpuPtr, gpuPtr, col.memoryUsage(in.size),
            cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        }

        JCuda.cudaStreamSynchronize(stream)

        // TODO in.free?

        out
      } {
        for (ptr <- gpuInputPtrs) {
          SparkEnv.get.cudaManager.deallocateGPUMemory(ptr)
        }
        for (ptr <- gpuOutputPtrs){
          SparkEnv.get.cudaManager.deallocateGPUMemory(ptr)
        }
      }
    } catch {
      case ex: Exception =>
        out.free
        throw ex
    }
  }

}
