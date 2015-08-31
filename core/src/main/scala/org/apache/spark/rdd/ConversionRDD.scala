package org.apache.spark.rdd

import scala.reflect.ClassTag

import org.apache.spark.{Partition, TaskContext, PartitionData}

private[spark] class ConversionRDD[T: ClassTag](
    prev: RDD[T]
  ) extends RDD[T](prev) {

  override def getPartitions: Array[Partition] =
    throw new UnsupportedOperationException("TODO") // TODO

  override def compute(split: Partition, context: TaskContext): PartitionData[T] =
    throw new UnsupportedOperationException("TODO") // TODO

}
