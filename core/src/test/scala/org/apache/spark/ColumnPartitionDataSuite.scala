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

import scala.reflect.ClassTag

class ColumnPartitionDataSuite extends SparkFunSuite with SharedSparkContext {

  def checkSerializationAndDeserialization[T: ClassTag](
      schema: ColumnPartitionSchema,
      input: Seq[T]): ColumnPartitionData[T] = {
    val data = new ColumnPartitionData[T](schema, input.size)
    assert(data.size == input.size)
    data.serialize(input.iterator)
    assert(data.buffers.forall(!_.hasRemaining))
    val output = data.deserialize().toArray
    assert(output.sameElements(input))
    assert(data.buffers.forall(!_.hasRemaining))
    data
  }

  test("Serializes single Int", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Int]
    val input = Array(42)
    val data = checkSerializationAndDeserialization(schema, input)
    data.rewind
    // checking if little endian
    val internalBuffer = data.buffers(0)
    assert(internalBuffer.get() == 42)
    assert(internalBuffer.get() == 0)
    assert(internalBuffer.get() == 0)
    assert(internalBuffer.get() == 0)
    data.free()
  }

  test("Serializes many Ints", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Int]
    val input = 1 to 42
    val data = checkSerializationAndDeserialization(schema, input)
    data.free()
  }

  test("Serializes object with no-arg constructor", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[TestClass]
    val input = Array(new TestClass)
    val data = checkSerializationAndDeserialization(schema, input)
    data.free()
  }

  test("Serializes nested case class objects", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Rectangle]
    val input = Array(
        Rectangle(Point(0, 0), Point(42, 42)),
        Rectangle(Point(2, 3), Point(8, 5)))
    val data = checkSerializationAndDeserialization(schema, input)
    data.free()
  }

}
