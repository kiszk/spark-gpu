package org.apache.spark

case class IteratedPartitionData[+T](
    iterator: Iterator[T]
  ) extends PartitionData[T] {
}
