import org.apache.commons.io.IOUtils

import org.apache.spark.{SparkConf, SparkContext, ColumnFormat, IteratorFormat}
import org.apache.spark.SparkContext._

object SparkGPUIntegrationTest {

  def main(args: Array[String]) {
    val conf = new SparkConf
    conf.setAppName("SparkGPUIntegrationTest")
    val sc = new SparkContext(conf)

    var ok = true

    val kernelResource = getClass.getClassLoader.getResourceAsStream("kernel.ptx")
    assert(kernelResource != null)
    val kernelData = IOUtils.toByteArray(kernelResource)

    sc.cudaManager.registerCUDAKernel(
      "multiplyBy2",
      "_Z11multiplyBy2PiS_l",
      Array("this"),
      Array("this"),
      kernelData)

    println("=== TEST 1 ===")
    val data = sc.parallelize(1 to 100000, 100)
      .convert(ColumnFormat)
      .mapUsingKernel((x: Int) => 2 * x, "multiplyBy2")
      .convert(IteratorFormat)
      .collect()
    println("Got data of length " + data.size)
    print(data.take(5).mkString(", ") + " ... ")
    println(data.takeRight(5).mkString(", "))
    if (!data.sameElements((1 to 100000).map(_ * 2))) {
      println("GOT WRONG DATA")
      ok = false
    }

    sc.stop()
    if (ok) {
      println("ALL TESTS PASSED")
    } else {
      println("THERE WERE FAILED TESTS")
      System.exit(1)
    }
  }

}
