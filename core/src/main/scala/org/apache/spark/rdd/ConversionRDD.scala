package org.apache.spark.rdd

import scala.reflect.ClassTag

import org.apache.spark.{Partition, TaskContext, PartitionData, IteratedPartitionData, ColumnPartitionData}

abstract class RDDFormat
case object IteratorRDDFormat extends RDDFormat
case object ColumnRDDFormat extends RDDFormat

private[spark] class ConversionRDD[T: ClassTag](
    prev: RDD[T],
    targetFormat: RDDFormat
  ) extends RDD[T](prev) {

  override def getPartitions: Array[Partition] =
    firstParent[T].partitions

  override def compute(split: Partition, context: TaskContext): PartitionData[T] = {
    val data = firstParent[T].partitionData(split, context)
    (data, targetFormat) match {
      // Cases where the format is already good
      case (it: IteratedPartitionData[T], IteratorRDDFormat) => it
      case (col: ColumnPartitionData[T], ColumnRDDFormat) => col

      // Converting from iterator-based format to column-based format
      case (IteratedPartitionData(iter), ColumnRDDFormat) =>
        throw new UnsupportedOperationException("TOOD") // TODO

      // Converting from column-based format to iterator-based format
      case (col: ColumnPartitionData[T], IteratorRDDFormat) =>
        IteratedPartitionData(col.iterator)
    }
  }

}
