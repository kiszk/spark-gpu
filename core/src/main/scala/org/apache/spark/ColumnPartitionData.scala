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

import org.apache.spark.unsafe.memory.MemoryBlock
import org.apache.spark.annotation.{DeveloperApi, Experimental}

import jcuda.Pointer

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import scala.language.existentials

// scalastyle:off no.finalize
@DeveloperApi
@Experimental
class ColumnPartitionData[T](
    val schema: ColumnPartitionSchema,
    val size: Int
  ) extends PartitionData[T] {

  private val pointers: Array[Pointer] = schema.columns.map { col =>
    SparkEnv.get.executorMemoryManager.allocatePointer(col.columnType.bytes * size)
  }

  private var freed = false

  val buffers: Array[ByteBuffer] = (pointers zip schema.columns).map { case (ptr, col) =>
    ptr.getByteBuffer(0, col.columnType.bytes * size)
  }

  def memoryUsage: Long = schema.columns.map(_.columnType.bytes * size).sum

  /**
   * Deallocate internal memory.
   */
  def free() {
    assert(!freed)
    pointers.foreach(SparkEnv.get.executorMemoryManager.freePointer(_))
    freed = true
  }

  /**
   * Finalizer method to free the memory if it was not freed yet for some reason.
   * Prints a warning
   */
  override def finalize() {
    if (!freed) {
      free()
      if (ColumnPartitionData.logger.isWarnEnabled()) {
        ColumnPartitionData.logger.warn("{}B of memory still not freed in finalizer.", memoryUsage);
      }
    }
  }

  def rewind {
    buffers.foreach(_.rewind)
  }

  // TODO allow for dropping specific columns if some kind of optimizer detected that they are not
  // needed
  def serialize(iter: Iterator[T]) {
    val getters = schema.getters
    rewind

    iter.foreach { obj =>
      for (((col, getter), buf) <- ((schema.columns zip getters) zip buffers)) {
        // TODO what should we do if sub-object is null?
        // TODO bulk put/get might be faster

        col.columnType match {
          case BYTE_COLUMN => buf.put(getter(obj).asInstanceOf[Byte])
          case SHORT_COLUMN => buf.putShort(getter(obj).asInstanceOf[Short])
          case INT_COLUMN => buf.putInt(getter(obj).asInstanceOf[Int])
          case LONG_COLUMN => buf.putLong(getter(obj).asInstanceOf[Long])
          case FLOAT_COLUMN => buf.putFloat(getter(obj).asInstanceOf[Float])
          case DOUBLE_COLUMN => buf.putDouble(getter(obj).asInstanceOf[Double])
        }
      }
    }
  }

  def deserialize(): Iterator[T] = {
    rewind

    if (schema.isPrimitive) {
      Iterator.continually {
        deserializeColumnValue(schema.columns(0).columnType, buffers(0)).asInstanceOf[T]
      } take size
    } else {
      // version of setters that creates objects that do not exist yet
      val setters: Array[(AnyRef, Any) => Unit] = {
        val mirror = ColumnPartitionSchema.mirror
        schema.columns.map { col =>
          val get: AnyRef => AnyRef = col.terms.dropRight(1).foldLeft(identity[AnyRef] _)
            { (r, term) => { (obj: AnyRef) =>
              val rf = mirror.reflect(obj).reflectField(term)
              rf.get match {
                case inner if inner != null => inner.asInstanceOf[AnyRef]
                case _ =>
                  val propCls = mirror.runtimeClass(term.typeSignature.typeSymbol.asClass)
                  val propVal = instantiateClass(propCls, obj)
                  rf.set(propVal)
                  propVal
              } } compose r
            }

          (obj: Any, value: Any) => mirror.reflect(get(obj.asInstanceOf[AnyRef]))
            .reflectField(col.terms.last).set(value)
        }
      }

      Iterator.continually {
        val obj = instantiateClass(schema.cls, null)

        for (((col, setter), buf) <- ((schema.columns zip setters) zip buffers)) {
          setter(obj, deserializeColumnValue(col.columnType, buf))
        }

        obj.asInstanceOf[T]
      } take size
    }
  }

  private[spark] def deserializeColumnValue(columnType: ColumnType, buf: ByteBuffer): Any = {
    columnType match {
      case BYTE_COLUMN => buf.get()
      case SHORT_COLUMN => buf.getShort()
      case INT_COLUMN => buf.getInt()
      case LONG_COLUMN => buf.getLong()
      case FLOAT_COLUMN => buf.getFloat()
      case DOUBLE_COLUMN => buf.getDouble()
    }
  }

  // taken from ClosureCleaner
  private[spark] def instantiateClass(
      cls: Class[_],
      enclosingObject: AnyRef): AnyRef = {
    // Use reflection to instantiate object without calling constructor
    val rf = sun.reflect.ReflectionFactory.getReflectionFactory()
    val parentCtor = classOf[java.lang.Object].getDeclaredConstructor()
    val newCtor = rf.newConstructorForSerialization(cls, parentCtor)
    val obj = newCtor.newInstance().asInstanceOf[AnyRef]
    if (enclosingObject != null) {
      val field = cls.getDeclaredField("$outer")
      field.setAccessible(true)
      field.set(obj, enclosingObject)
    }
    obj
  }

  override def iterator: Iterator[T] =
    throw new UnsupportedOperationException("TODO") // TODO

}
// scalastyle:on no.finalize

object ColumnPartitionData {

  private final val logger: Logger = LoggerFactory.getLogger(classOf[ColumnPartitionData[_]])

}
