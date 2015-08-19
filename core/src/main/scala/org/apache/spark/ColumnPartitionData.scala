package org.apache.spark

import org.apache.spark.unsafe.memory.MemoryBlock

case class ColumnPartitionData[+T](
    columns: Array[MemoryBlock]
  ) extends PartitionData[T] {

  def memoryUsage: Long = columns.map(_.size).sum

}
