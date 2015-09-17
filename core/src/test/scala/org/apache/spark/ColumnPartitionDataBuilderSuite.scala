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

class ColumnPartitionDataBuilderSuite extends SparkFunSuite with SharedSparkContext {

  test("Creates uninitialized ColumnPartitionData") {
    val data = ColumnPartitionDataBuilder.build[Byte](10000)
    assert(data.schema.columns.length == 1)
    assert(data.size == 10000)
    data.free()
  }

  test("Creates ColumnPartitionData for a single Int", GPUTest) {
    val input = Array(42)
    val data = ColumnPartitionDataBuilder.build[Int](1)
    assert(data.schema.columns.length == 1)
    assert(data.size == 1)
    data.serialize(input.iterator)
    val output = data.deserialize().toIndexedSeq
    assert(output.sameElements(input.toIndexedSeq))
    data.free()
  }

  test("Creates ColumnPartitionData from a sequence of case classes") {
    val input = Array(
        Rectangle(Point(0, 0), Point(42, 42)),
        Rectangle(Point(2, 3), Point(8, 5)))
    val data = ColumnPartitionDataBuilder.build(input)
    assert(data.schema.cls != null)
    val output = data.deserialize().toIndexedSeq
    assert(output.sameElements(input.toIndexedSeq))
    data.free()
  }

  test("Creates a ColumnPartitionData from an iterator without length information") {
    val input = Array(
        Rectangle(Point(1, 2), Point(3, 4)),
        Rectangle(Point(5, 6), Point(7, 8)),
        Rectangle(Point(9, 10), Point(11, 12)),
        Rectangle(Point(13, 14), Point(15, 16)))
    val inputIter = input.iterator
    val data = ColumnPartitionDataBuilder.build(inputIter)
    assert(data.schema.cls != null)
    val output = data.deserialize().toIndexedSeq
    assert(output.sameElements(input))
    data.free()
  }

  test("Creates a ColumnPartitionData from an iterator with shorter length information") {
    val input = Array(
        Rectangle(Point(1, 2), Point(3, 4)),
        Rectangle(Point(5, 6), Point(7, 8)),
        Rectangle(Point(9, 10), Point(11, 12)),
        Rectangle(Point(13, 14), Point(15, 16)))
    val inputIter = input.iterator
    val data = ColumnPartitionDataBuilder.build(inputIter, 3)
    assert(data.schema.cls != null)
    val output = data.deserialize().toIndexedSeq
    assert(output.sameElements(input.take(3)))
    data.free()
  }

}
