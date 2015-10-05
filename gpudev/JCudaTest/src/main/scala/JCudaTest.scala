import jcuda.runtime.JCuda
import jcuda.driver.JCudaDriver

object JCudaTest {
  def main(args: Array[String]) {
    JCudaDriver.setExceptionsEnabled(true)
 
    val memInfo = Array.fill(2)(new Array[Long](1))
    JCuda.cudaMemGetInfo(memInfo(0), memInfo(1))
    val freeMem = memInfo(0)(0)
    val totalMem = memInfo(1)(0)

    println(s"Free: $freeMem Total: $totalMem")
  }
}
