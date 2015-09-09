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

package org.apache.spark

import scala.reflect.runtime.universe.Type
import scala.reflect.runtime.universe.TypeTag

import org.apache.spark.annotation.DeveloperApi

import java.nio.ByteBuffer

@DeveloperApi
object ColumnPartitionDataBuilder {

  /**
   * Creates ColumnPartitionData with uninitialized data for given type and size.
   */
  def build[T: TypeTag](n: Long): ColumnPartitionData[T] = {
    build(ColumnPartitionSchema.localTypeOf[T], n).asInstanceOf[ColumnPartitionData[T]]
  }

  /**
   * Creates ColumnPartitionData with uninitialized data for given type and size.
   */
  def build(tpe: Type, n: Long): ColumnPartitionData[_] = {
    val schema = ColumnPartitionSchema.schemaForType(tpe)
    new ColumnPartitionData(schema, n)
  }

  /**
   * Converts an iterator to ColumnPartitionData.
   * Knowing the input size is not required, but will prevent materialization of the whole
   * data before conversion. Passing negative size means that it is to be automatically computed.
   */
  def build[T: TypeTag](it: Iterator[T], size: Long = -1): ColumnPartitionData[T] = {
    if (size >= 0) {
      val col = build[T](size)
      col.serialize(it)
      col
    } else {
      build(it.toIndexedSeq)
    }
  }

  /**
   * Converts a sequence into ColumnPartitionData.
   */
  def build[T: TypeTag](seq: Seq[T]): ColumnPartitionData[T] =
    build(seq.iterator, seq.size)

}
