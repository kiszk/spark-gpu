/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.rdd

import org.apache.spark.storage.RDDBlockId

import scala.reflect.ClassTag

import org.apache.spark.{Partition, TaskContext, PartitionData, IteratorPartitionData,
  ColumnPartitionData, PartitionFormat, IteratorFormat, SparkException}

/**
 * An RDD that applies the provided function to every partition of the parent RDD.
 */
private[spark] class MapPartitionsRDD[U: ClassTag, T: ClassTag](
    prev: RDD[T],
    f: (TaskContext, Int, Iterator[T]) => Iterator[U],  // (TaskContext, partition index, iterator)
    preservesPartitioning: Boolean = false,
    extfunc: Option[ExternalFunction] = None,
    outputArraySizes: Seq[Long] = null,
    inputFreeVariables: Seq[Any] = null)
  extends RDD[U](prev) {

  override val partitioner = if (preservesPartitioning) firstParent[T].partitioner else None

  override def getPartitions: Array[Partition] = firstParent[T].partitions

  override def compute(split: Partition, context: TaskContext): Iterator[U] = {
    throw new SparkException("We do not implement compute since computePartition is implemented.")
  }

  override def computePartition(split: Partition, context: TaskContext): PartitionData[U] = {
    (firstParent[T].partitionData(split, context), extfunc) match {
      // computing column-based partition on GPU
      case (col: ColumnPartitionData[T], Some(extfun)) =>
        extfun.run(col, None, outputArraySizes,
                        inputFreeVariables,Some(RDDBlockId(id, split.index)),gpuCache)
      // computing  iterator-based partition on CPU
      case (data, _) =>
        IteratorPartitionData(f(context, split.index, data.iterator))
    }
  }
}
