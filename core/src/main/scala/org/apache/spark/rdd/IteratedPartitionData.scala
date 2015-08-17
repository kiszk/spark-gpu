package org.apache.spark.rdd

case class IteratedPartitionData[T](
    iterator: Iterator[T]
  ) extends PartitionData[T] {
}
