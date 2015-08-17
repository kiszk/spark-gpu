package org.apache.spark.rdd

case class ColumnPartitionData[T](
    column: Array[T]
  ) extends PartitionData[T] {
}
