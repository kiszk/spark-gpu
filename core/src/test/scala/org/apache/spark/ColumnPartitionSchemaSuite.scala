package org.apache.spark

import org.apache.spark._

import java.nio.ByteBuffer
import java.nio.ByteOrder

class TestClass {
  val x: Int = 1
  private val y: Double = 2
  var p: Short = 12
  private var q: Long = 42
  
  override def equals(obj: Any) = {
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
  test("SerializesSingleInt", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Int]
    val input = Array(42)
    assert(schema.columns.length == 1)
    assert(schema.isPrimitive)
    assert(schema.columns(0).columnType == INT_COLUMN)
    val buf = ByteBuffer.allocate(schema.columns(0).columnType.bytes).order(ByteOrder.LITTLE_ENDIAN)
    schema.serialize(input.iterator, Array(buf))
    assert(!buf.hasRemaining)
    assert(buf.array.sameElements(Array(42, 0, 0, 0)))
    buf.rewind
    val output = schema.deserialize(Array(buf), 1).toArray
    assert(!buf.hasRemaining)
    assert(output.sameElements(input))
  }

  test("SerializesManyInts", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Int]
    val input = 1 to 42
    assert(schema.columns.length == 1)
    assert(schema.isPrimitive)
    assert(schema.columns(0).columnType == INT_COLUMN)
    val buf = ByteBuffer.allocate(42 * schema.columns(0).columnType.bytes)
    schema.serialize(input.iterator, Array(buf))
    assert(!buf.hasRemaining)
    buf.rewind
    val output = schema.deserialize(Array(buf), 42).toArray
    assert(!buf.hasRemaining)
    assert(output.sameElements(input))
  }

  test("SerializesObjectWithNoArgConstructor", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[TestClass]
    val input = Array(new TestClass)
    assert(schema.columns.length == 4)
    assert(schema.columns.map(_.columnType).toSet == Set(INT_COLUMN, DOUBLE_COLUMN, SHORT_COLUMN, LONG_COLUMN))
    assert(!schema.isPrimitive)
    val bufs = schema.columns.map(col => ByteBuffer.allocate(col.columnType.bytes))
    schema.serialize(input.iterator, bufs)
    bufs.foreach(buf => assert(!buf.hasRemaining))
    bufs.foreach(_.rewind)
    val output = schema.deserialize(bufs, 1).toArray
    bufs.foreach(buf => assert(!buf.hasRemaining))
    assert(output.sameElements(input))
  }

  test("SerializeNestedCaseClassObjects", GPUTest) {
    val schema = ColumnPartitionSchema.schemaFor[Rectangle]
    val input = Array(
        Rectangle(Point(0, 0), Point(42, 42)),
        Rectangle(Point(2, 3), Point(8, 5)))
    assert(schema.columns.length == 4)
    schema.columns.foreach(col => assert(col.columnType == INT_COLUMN))
    assert(!schema.isPrimitive)
    val bufs = schema.columns.map(col => ByteBuffer.allocate(2 * col.columnType.bytes))
    schema.serialize(input.iterator, bufs)
    bufs.foreach(buf => assert(!buf.hasRemaining))
    bufs.foreach(_.rewind)
    val output = schema.deserialize(bufs, 2).toArray
    bufs.foreach(buf => assert(!buf.hasRemaining))
    assert(output.sameElements(input))
  }
}
