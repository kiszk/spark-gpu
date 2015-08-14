package org.apache.spark.storage

case class IteratedPartitionData[T](
    iterator: Iterator[T]
  ) extends PartitionData[T] {
}
