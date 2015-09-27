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
import scala.math._

import org.apache.spark.{Partition, TaskContext, PartitionFormat, PartitionData}

private[spark] class ConvertRDD[T: ClassTag](
    prev: RDD[T],
    targetFormat: PartitionFormat,
    ratio: Double = 1.0
  ) extends RDD[T](prev) {

  override def getPartitions: Array[Partition] =
    firstParent[T].partitions

  override def computePartition(split: Partition, context: TaskContext): PartitionData[T] = {
    // ceil(n * ratio) - how many partitions should be converted in first n partitions to maintain
    // the ratio
    // substract value for n-1 from that and you'll get either 0 or 1 - if you have to convert the
    // current partition
    // this maintains the ratio with good accuracy +- 1 for every interval of partition indexes
    // TODO this works only once per complete conversion, applying this twice will not yield
    // expected results
    if (ceil((split.index + 1) * ratio).toInt - ceil(split.index * ratio) > 0) {
      firstParent[T].partitionData(split, context).convert(targetFormat)
    } else {
      firstParent[T].partitionData(split, context)
    }
  }

}
