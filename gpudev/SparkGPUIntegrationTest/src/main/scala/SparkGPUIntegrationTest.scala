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

    val dimensions = (size: Long, stage: Int) => stage match {
      case 0 => (4, 32)
      case 1 => (1, 1)
    }
    sc.cudaManager.registerCUDAKernel(
      "sum",
      "_Z3sumPiS_lii",
      Array("this"),
      Array("this"),
      kernelData,
      Seq[AnyVal](),
      Some(2),
      Some(dimensions))

    println("=== TEST 1 ===")
    println("= map =")
    val data = sc.parallelize(1 to 100000, 10)
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

    println("=== TEST 2 ===")
    println("= map + reduce =")
    val sum = sc.parallelize(1 to 10)
      .convert(ColumnFormat)
      .mapUsingKernel((x: Int) => 2 * x, "multiplyBy2")
      .reduceUsingKernel((x: Int, y: Int) => x + y, "sum")
    println(s"Got result: $sum")
    if (sum != 100010000) {
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
