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

import java.nio.ByteBuffer
import java.nio.ByteOrder

class TestClass {
  val x: Int = 1
  private val y: Double = 2
  var p: Short = 12
  private var q: Long = 42

  override def equals(obj: Any): Boolean = {
    obj match {
      case o: TestClass => (o.x == x) && (o.y == y) && (o.p == p) && (o.q == q)
      case _ => false
    }
  }
}

// Note: it doesn't have a no-arg constructor
case class Point(x: Int, y: Int)
case class Rectangle(topLeft: Point, bottomRight: Point)

class ColumnPartitionSchemaSuite extends SparkFunSuite with SharedSparkContext {

  test("Schema for Int", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Int]
    assert(schema.columns.length == 1)
    assert(schema.isPrimitive)
    assert(schema.columns(0).columnType == INT_COLUMN)
    assert(schema.columns(0).prettyAccessor == "this")
  }

  test("Schema for class with no-arg constructor", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[TestClass]
    assert(schema.columns.length == 4)
    assert(schema.columns.map(_.columnType).toSet ==
        Set(INT_COLUMN, DOUBLE_COLUMN, SHORT_COLUMN, LONG_COLUMN))
    assert(!schema.isPrimitive)
    assert(schema.cls != null)
    assert(schema.columns.map(_.prettyAccessor).toSet ==
      Set("this.x", "this.y", "this.p", "this.q"))
  }

  test("Schema for case class", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Rectangle]
    assert(schema.columns.length == 4)
    schema.columns.foreach(col => assert(col.columnType == INT_COLUMN))
    assert(!schema.isPrimitive)
    assert(schema.cls != null)
    assert(schema.columns.map(_.prettyAccessor).toSet ==
      Set("this.topLeft.x", "this.topLeft.y", "this.bottomRight.x", "this.bottomRight.y"))
  }

  test("Getters work for Int", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Int]
    val getters = schema.getters
    assert(getters.size == 1)
    assert(getters(0)(42) == 42)
  }

  test("Getters work for case class", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Rectangle]
    val getters = schema.getters
    assert(getters.size == 4)
    val rect = Rectangle(Point(0, 1), Point(2, 3))
    assert(getters.map(get => get(rect).asInstanceOf[Int]).toSet == Set(0, 1, 2, 3))
  }

  test("Setters work for case class", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Rectangle]
    val setters = schema.setters
    assert(setters.size == 4)
    val rect = Rectangle(Point(0, 1), Point(2, 3))
    setters.foreach(set => set(rect, 42))
    assert(rect == Rectangle(Point(42, 42), Point(42, 42)))
  }

}
