package org.apache.spark

import org.scalatest.Tag

private[spark] object SlowTest extends Tag("org.apache.sparktest.SlowTest")
private[spark] object GPUTest extends Tag("org.apache.sparktest.GPUTest")
private[spark] object PPCIBMJDKFailingTest extends Tag("org.apache.sparktest.PPCIBMJDKFailingTest")
