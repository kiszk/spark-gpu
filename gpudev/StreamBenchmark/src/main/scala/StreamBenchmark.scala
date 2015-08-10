import jcuda.Pointer
import jcuda.driver.JCudaDriver
import jcuda.driver.CUmodule
import jcuda.driver.CUfunction
import jcuda.driver.CUstream
import jcuda.driver.CUdevice
import jcuda.driver.CUcontext
import jcuda.runtime.JCuda
import jcuda.runtime.cudaStream_t
import jcuda.runtime.cudaMemcpyKind

class StreamThread(
    dataSize: Long,
    cpuDataPtr: Pointer,
    iters: Int,
    expectedResult: Int) extends Runnable {

  def run() {
    val dataBytes = dataSize * 4

    val device = new CUdevice
    JCudaDriver.cuDeviceGet(device, 0)
    val context = new CUcontext
    JCudaDriver.cuCtxCreate(context, 0, device)

    val module = new CUmodule
    JCudaDriver.cuModuleLoadData(module, StreamBenchmark.kernelCode)
    val function = new CUfunction
    JCudaDriver.cuModuleGetFunction(function, module, StreamBenchmark.kernelName)

    val gpuDataPtr = new Pointer
    JCuda.cudaMalloc(gpuDataPtr, dataBytes)

    val stream = new cudaStream_t
    JCuda.cudaStreamCreate(stream)

    for (k <- 1 to iters) {
      JCuda.cudaMemcpyAsync(gpuDataPtr, cpuDataPtr, dataBytes,
          cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

      val params1 = Pointer.to(
          Pointer.to(gpuDataPtr),
          Pointer.to(Array(dataSize)))
      JCudaDriver.cuLaunchKernel(
          function,
          StreamBenchmark.gpuBlocks, 1, 1,
          StreamBenchmark.gpuBlockSize, 1, 1,
          0,
          new CUstream(stream),
          params1, null)

      val params2 = Pointer.to(
          Pointer.to(gpuDataPtr),
          Pointer.to(Array(StreamBenchmark.gpuThreads)))
      JCudaDriver.cuLaunchKernel(
          function,
          1, 1, 1,
          1, 1, 1,
          0,
          new CUstream(stream),
          params2, null)

      val result = new Array[Int](1)
      val resultPtr = Pointer.to(result)

      JCuda.cudaMemcpyAsync(resultPtr, gpuDataPtr, 4,
          cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

      JCuda.cudaStreamSynchronize(stream)

      if (result(0) != expectedResult) {
        throw new Exception(s"Got result ${result(0)}, but expected $expectedResult")
      }
    }

    JCuda.cudaStreamDestroy(stream)

    JCuda.cudaFree(gpuDataPtr)
  }

}

object StreamBenchmark {
  val gpuBlocks = 8
  val gpuBlockSize = 1024
  val gpuThreads = gpuBlocks * gpuBlockSize
  val kernelName = "_Z3sumPim"
  def kernelPath = "/kernel.ptx"

  def kernelCode() : Array[Byte] = {
    val input = getClass.getResourceAsStream(kernelPath)
    Stream.continually(input.read).takeWhile(s => s != -1).map(_.toByte).toArray
  }

  def bench(f: => Unit) : Long = {
    val before = System.nanoTime
    f
    val after = System.nanoTime
    (after - before) / 1000000
  }

  def runStreams(dataSize: Long, iters: Int, streams: Int) {
    val dataBytes = dataSize * 4

    val cpuDataPtr = new Pointer
    // it might be more efficient to use cudaHostAllocWriteCombined for reduction kernel
    // (though not for map kernel)
    JCuda.cudaHostAlloc(cpuDataPtr, dataBytes, JCuda.cudaHostAllocDefault)
    val buf = cpuDataPtr.getByteBuffer(0, dataBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asIntBuffer
    for (i <- 1l to dataSize) {
      buf.put(if (i % 42 == 0) 2 else 0)
    }
    val expectedResult = (dataSize / 42 * 2).toInt

    val time = bench {
      val threads = Array.fill[Thread](streams)(new Thread(
          new StreamThread(dataSize, cpuDataPtr, iters, expectedResult)))
      threads.map(_.start)
      threads.map(_.join)
    }

    println(s"Data size: $dataSize (${dataSize / gpuThreads} / thread)\n")
    println(s"GPU threads: $gpuBlocks x $gpuBlockSize = $gpuThreads")
    println(s"Time: ${time}ms")
  }

  private def usageQuit() {
    println("Usage: <StreamBenchmark invocation> dataSize count streams")
    println("Program runs a simple kernel in `streams` streams `count` times.")
    println("Each run consists of copying `dataSize` MB from CPU to GPU, running a kernel " +
        "and finally copies few bytes of result to CPU memory.")
    println("Each stream is ran from a separate thread.")

    System.exit(1)
  }

  def main(args: Array[String]) {
    println("CUDA stream benchmark")
    if (args.length != 3) {
      usageQuit();
    }

    JCudaDriver.setExceptionsEnabled(true)
    JCudaDriver.cuInit(0)

    val memInfo = Array.fill(2)(new Array[Long](1))
    JCuda.cudaMemGetInfo(memInfo(0), memInfo(1))
    val freeMem = memInfo(0)(0)
    val totalMem = memInfo(1)(0)
    
    val dataSize = 1024 * 1024 / 4 * args(0).toLong
    val iters = args(1).toInt
    val streams = args(2).toInt

    val neededMem = (streams * dataSize * 4)

    println(s"Free GPU memory: ${freeMem >> 20}MB Total: ${totalMem >> 20}MB Needed: ${neededMem >> 20}MB")

    runStreams(dataSize, iters, streams)

    JCuda.cudaDeviceReset();
  }
}
