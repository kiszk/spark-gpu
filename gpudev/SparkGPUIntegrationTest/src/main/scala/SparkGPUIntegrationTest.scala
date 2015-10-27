import org.apache.commons.io.IOUtils

import org.apache.spark.{SparkConf, SparkContext, ColumnFormat, IteratorFormat}
import org.apache.spark.SparkContext._

import org.apache.log4j.Logger
import org.apache.log4j.Level

object SparkGPUIntegrationTest {

  def main(args: Array[String]) {
    val conf = new SparkConf
    conf.setAppName("SparkGPUIntegrationTest")
    val sc = new SparkContext(conf)

    Logger.getLogger("org.apache.spark").setLevel(Level.DEBUG);
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF);

    var ok = true

    val kernelResource = getClass.getClassLoader.getResourceAsStream("kernel.ptx")
    assert(kernelResource != null)
    val kernelData = IOUtils.toByteArray(kernelResource)

    val mapKernel = new CUDAKernel(
      "multiplyBy2",
      "_Z11multiplyBy2PiS_l",
      Array("this"),
      Array("this"),
      kernelData)

    val stages = (size: Long) => 2
    val dimensions = (size: Long, stage: Int) => stage match {
      case 0 => (4, 32)
      case 1 => (1, 1)
    }
    val reduceKernel = new CUDAKernel(
      "sum",
      "_Z3sumPiS_lii",
      Array("this"),
      Array("this"),
      kernelData, CUDAKernelPtxResourceKind,
      Seq[AnyVal](),
      Some(stages),
      Some(dimensions))

    {
      println("=== TEST 1 ===")
      println("= map =")
      val n = 100000
      val data = sc.parallelize(1 to n, 10)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, "multiplyBy2")
        .collect()
      println("Got data of length " + data.size)
      print(data.take(5).mkString(", ") + " ... ")
      println(data.takeRight(5).mkString(", "))
      if (!data.sameElements((1 to n).map(_ * 2))) {
        println("GOT WRONG DATA")
        ok = false
      }
    }

    {
      println("=== TEST 2 ===")
      println("= reduce =")
      val n = 10000
      val result = sc.parallelize(1 to n, 10)
        .convert(ColumnFormat)
        .reduceUsingKernel((x: Int, y: Int) => x + y, "sum")
      println(s"Got result: $result")
      if (result != n * (n + 1) / 2) {
        println("GOT WRONG DATA")
        ok = false
      }
    }

    {
    println("=== TEST 3 ===")
    println("= map + map =")
    val n = 100000
    val data = sc.parallelize(1 to n, 10)
      .convert(ColumnFormat)
      .mapUsingKernel((x: Int) => 2 * x, "multiplyBy2")
      .mapUsingKernel((x: Int) => 2 * x, "multiplyBy2")
      .collect()
    println("Got data of length " + data.size)
    print(data.take(5).mkString(", ") + " ... ")
    println(data.takeRight(5).mkString(", "))
    if (!data.sameElements((1 to n).map(_ * 4))) {
      println("GOT WRONG DATA")
      ok = false
    }
    }

    {
      println("=== TEST 3 ===")
      println("= map + reduce =")
      val n = 10000
      val result = sc.parallelize(1 to n, 10)
        .convert(ColumnFormat)
        .mapUsingKernel((x: Int) => 2 * x, "multiplyBy2")
        .reduceUsingKernel((x: Int, y: Int) => x + y, "sum")
      println(s"Got result: $result")
      if (result != n * (n + 1)) {
        println("GOT WRONG DATA")
        ok = false
      }
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
