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

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.Type
import scala.reflect.runtime.universe.TypeTag
import scala.reflect.runtime.universe.TermSymbol

import org.apache.spark.util.Utils

// Some code taken from org.apache.spark.sql.catalyst.ScalaReflection

object ColumnPartitionSchema {

  // Since we are creating a runtime mirror usign the class loader of current thread,
  // we need to use def at here. So, every time we call mirror, it is using the
  // class loader of the current thread.
  private[spark] def mirror: universe.Mirror =
    universe.runtimeMirror(Thread.currentThread().getContextClassLoader)

  var onlyLoadableClassesSupported: Boolean = false

  private[spark] def localTypeOf[T: TypeTag] = universe.typeTag[T].in(mirror).tpe

  def schemaFor[T: TypeTag]: ColumnPartitionSchema =
    schemaForType(localTypeOf[T])

  def schemaForType(tpe: Type): ColumnPartitionSchema = {
    tpe match {
      // 8-bit signed BE
      case t if t <:< localTypeOf[Byte] => primitiveColumnPartitionSchema(1, BYTE_COLUMN)
      // 16-bit signed BE
      case t if t <:< localTypeOf[Short] => primitiveColumnPartitionSchema(2, SHORT_COLUMN)
      // 32-bit signed BE
      case t if t <:< localTypeOf[Int] => primitiveColumnPartitionSchema(4, INT_COLUMN)
      // 64-bit signed BE
      case t if t <:< localTypeOf[Long] => primitiveColumnPartitionSchema(8, LONG_COLUMN)
      // 32-bit single-precision IEEE 754 floating point
      case t if t <:< localTypeOf[Float] => primitiveColumnPartitionSchema(4, FLOAT_COLUMN)
      // 64-bit double-precision IEEE 754 floating point
      case t if t <:< localTypeOf[Double] => primitiveColumnPartitionSchema(8, DOUBLE_COLUMN)
      // TODO boolean - it does not have specified size
      // TODO char
      // TODO string - along with special storage space
      // TODO array
      // TODO option
      // TODO protection from cycles
      // TODO caching schemas for classes
      // TODO make it work with nested classes
      // Generic object
      case t if !onlyLoadableClassesSupported ||
          Utils.classIsLoadable(t.typeSymbol.asClass.fullName) => {
        val valVarMembers = t.erasure.members.view
          .filter(p => !p.isMethod && p.isTerm).map(_.asTerm)
          .filter(p => p.isVar || p.isVal)

        valVarMembers.foreach { p =>
          // TODO more checks
          // is final okay?
          if (p.isStatic) throw new UnsupportedOperationException(
              s"Column schema with static field ${p.fullName} not supported")
        }

        val columns = valVarMembers.flatMap { term =>
          schemaForType(term.typeSignature).columns.map { schema =>
            new ColumnSchema(
              schema.columnType,
              term +: schema.terms)
          }
        }

        new ColumnPartitionSchema(columns.toArray, mirror.runtimeClass(tpe.typeSymbol.asClass))
      }
      case other =>
        throw new UnsupportedOperationException(s"Column schema for type $other not supported")
    }
  }

  private[spark] def primitiveColumnPartitionSchema(bytes: Int, columnType: ColumnType) =
    new ColumnPartitionSchema(Array(new ColumnSchema(columnType)), null)

}

class ColumnPartitionSchema(
    val columns: Array[ColumnSchema],
    val cls: Class[_]) {

  def isPrimitive: Boolean = columns.size == 1 && columns(0).terms.isEmpty

  def getters: Array[Any => Any] = {
    assert(!isPrimitive)
    val mirror = ColumnPartitionSchema.mirror
    columns.map { col =>
      col.terms.foldLeft(identity[Any] _)((r, term) =>
          ((obj: Any) => mirror.reflect(obj).reflectField(term).get) compose r)
    }
  }

  def setters: Array[(Any, Any) => Unit] = {
    assert(!isPrimitive)
    val mirror = ColumnPartitionSchema.mirror
    columns.map { col =>
      val getOuter = col.terms.dropRight(1).foldLeft(identity[Any] _)((r, term) =>
          ((obj: Any) => mirror.reflect(obj).reflectField(term).get) compose r)

      (obj: Any, value: Any) =>
        mirror.reflect(getOuter(obj)).reflectField(col.terms.last).set(value)
    }
  }

}

abstract class ColumnType {
  /** How many bytes does a single property take. */
  val bytes: Int
}

case object BYTE_COLUMN extends ColumnType {
  val bytes = 1
}

case object SHORT_COLUMN extends ColumnType {
  val bytes = 2
}

case object INT_COLUMN extends ColumnType {
  val bytes = 4
}

case object LONG_COLUMN extends ColumnType {
  val bytes = 8
}

case object FLOAT_COLUMN extends ColumnType {
  val bytes = 4
}

case object DOUBLE_COLUMN extends ColumnType {
  val bytes = 8
}

/**
 * A column is one basic property (primitive, String, etc.).
 */
class ColumnSchema(
    /** Type of the property. Is null when the whole object is a primitive. */
    val columnType: ColumnType,
    /** Scala terms with property name and other information */
    val terms: Vector[TermSymbol] = Vector[TermSymbol]()) {

//  def name: Array[String] = terms.map(_.name)
}
