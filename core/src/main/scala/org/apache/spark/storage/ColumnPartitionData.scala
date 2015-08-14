package org.apache.spark.storage

case class ColumnPartitionData[T](
    column: Array[T]
  ) extends PartitionData[T] {
}
