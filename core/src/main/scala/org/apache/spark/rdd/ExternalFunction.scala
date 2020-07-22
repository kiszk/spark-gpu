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

import org.apache.spark.storage.BlockId

import scala.reflect.ClassTag

import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.ColumnPartitionData

/**
 * :: Experimental ::
 * An abstract class to define hand-written external function for experiments
 * of column-based format RDD
 */

@Experimental
abstract class ExternalFunction extends Serializable {
  private[spark] def run[T, U: ClassTag](in: ColumnPartitionData[T],
      outputSize: Option[Long] = None,
      outputArraySizes: Seq[Long] = null,
      inputFreeVariables: Seq[Any] = null,
      blockId : Option[BlockId] = None): ColumnPartitionData[U]
}
