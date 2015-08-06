import com.ibm.cuda.CudaDevice
import com.ibm.cuda.CudaModule
import com.ibm.cuda.CudaKernel
import com.ibm.cuda.CudaGrid
import com.ibm.cuda.CudaBuffer

object CUDA4JTest {
  def main(args: Array[String]) {
    val device = new CudaDevice(0)
    val freeMem = device.getFreeMemory()
    println(s"Bytes free: $freeMem")

    val module = new CudaModule(device, getClass.getResourceAsStream("/kernel.cubin"))
    val kernel = new CudaKernel(module, "_Z6kernelPii")
    val grid = new CudaGrid(1, 32)

    val cpuBuf = Array.fill[Int](128)(0)

    val gpuBuf = new CudaBuffer(device, 4 * cpuBuf.length)

    val input = cpuBuf.mkString(", ")
    println(s"Input: $input")

    try {
      gpuBuf.copyFrom(cpuBuf, 0, cpuBuf.length)

      kernel.launch(grid, new CudaKernel.Parameters(gpuBuf, cpuBuf.length : Integer))

      gpuBuf.copyTo(cpuBuf, 0, cpuBuf.length)
    } finally {
      gpuBuf.close()
    }

    val output = cpuBuf.mkString(", ")
    println(s"Output: $output")
  }
}
