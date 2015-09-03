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

import scala.reflect.ClassTag

import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.{PartitionData, IteratedPartitionData, ColumnPartitionData}

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
