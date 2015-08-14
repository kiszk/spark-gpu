package org.apache.spark

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.Type
import scala.reflect.runtime.universe.TypeTag
import scala.reflect.runtime.universe.TermSymbol

import java.nio.ByteBuffer

import org.apache.spark.util.Utils
import org.apache.spark.util.ClosureCleaner

// Some code taken from org.apache.spark.sql.catalyst.ScalaReflection

object ColumnPartitionSchema {

  // Since we are creating a runtime mirror usign the class loader of current thread,
  // we need to use def at here. So, every time we call mirror, it is using the
  // class loader of the current thread.
  def mirror: universe.Mirror =
    universe.runtimeMirror(Thread.currentThread().getContextClassLoader)

  var onlyLoadableClassesSupported: Boolean = false

  private def localTypeOf[T: TypeTag] = universe.typeTag[T].in(mirror).tpe

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
      // Generic object
      case t if !onlyLoadableClassesSupported || Utils.classIsLoadable(t.typeSymbol.asClass.fullName) => {
        val valVarMembers = t.erasure.members.view
          .filter(p => !p.isMethod && p.isTerm).map(_.asTerm)
          .filter(p => p.isVar || p.isVal)

        valVarMembers.map { p =>
          // TODO more checks
          // is final okay?
          if (p.isStatic) throw new UnsupportedOperationException(s"Column schema with static field ${p.fullName} not supported")
        }

        val columns = valVarMembers.flatMap { term =>
          schemaForType(term.typeSignature).columns.map { schema =>
            new ColumnSchema(
              schema.columnType,
              schema.bytes,
              term +: schema.terms)
          }
        }

        new ColumnPartitionSchema(columns.toArray, mirror.runtimeClass(tpe.typeSymbol.asClass))
      }
      case other =>
        throw new UnsupportedOperationException(s"Column schema for type $other not supported")
    }
  }

  def primitiveColumnPartitionSchema(bytes: Int, columnType: ColumnType) =
    new ColumnPartitionSchema(Array(new ColumnSchema(columnType, bytes)), null)

}

class ColumnPartitionSchema(
    val columns: Array[ColumnSchema],
    val cls: Class[_]) {

  def isPrimitive = columns.size == 1 && columns(0).name.isEmpty

  def serializeObject(obj: Any, columnBuffers: Seq[ByteBuffer]) {
    val mirror = ColumnPartitionSchema.mirror

    for ((col, buf) <- (columns zip columnBuffers)) {
      val get = col.terms.foldLeft(identity[Any] _)((r, term) =>
          ((obj: Any) => mirror.reflect(obj).reflectField(term).get) compose r)
      // TODO what should we do if sub-object is null?

      col.columnType match {
        case BYTE_COLUMN => buf.put(get(obj).asInstanceOf[Byte])
        case SHORT_COLUMN => buf.putShort(get(obj).asInstanceOf[Short])
        case INT_COLUMN => buf.putInt(get(obj).asInstanceOf[Int])
        case LONG_COLUMN => buf.putLong(get(obj).asInstanceOf[Long])
        case FLOAT_COLUMN => buf.putFloat(get(obj).asInstanceOf[Float])
        case DOUBLE_COLUMN => buf.putDouble(get(obj).asInstanceOf[Double])
      }
    }
  }
  // TODO method to deserialize Iterator[Any] objects and not repeat creating get function

  def deserializeObject(columnBuffers: Seq[ByteBuffer]): Any = {
    if (isPrimitive) {
      deserializeColumnValue(columns(0).columnType, columnBuffers(0))
    } else {
      val obj = ClosureCleaner.instantiateClass(cls, null)

      val mirror = ColumnPartitionSchema.mirror

      for ((col, buf) <- (columns zip columnBuffers)) {
        val get = col.terms.dropRight(1).foldLeft(identity[Any] _)((r, term) => {(obj: Any) =>
              val rf = mirror.reflect(obj).reflectField(term)
              rf.get match {
                case inner if inner != null => inner
                case _ => {
                  val propCls = mirror.runtimeClass(term.typeSignature.typeSymbol.asClass)
                  val propVal = ClosureCleaner.instantiateClass(propCls, null)
                  rf.set(propVal)
                  propVal
                }
              }
            } compose r)

        mirror.reflect(get(obj)).reflectField(col.terms.last).set(deserializeColumnValue(col.columnType, buf))
      }
      
      obj
    }
  }

  def deserializeColumnValue(columnType: ColumnType, buf: ByteBuffer): Any = {
    columnType match {
      case BYTE_COLUMN => buf.get()
      case SHORT_COLUMN => buf.getShort()
      case INT_COLUMN => buf.getInt()
      case LONG_COLUMN => buf.getLong()
      case FLOAT_COLUMN => buf.getFloat()
      case DOUBLE_COLUMN => buf.getDouble()
    }
  }

}

abstract class ColumnType
case object BYTE_COLUMN extends ColumnType
case object SHORT_COLUMN extends ColumnType
case object INT_COLUMN extends ColumnType
case object LONG_COLUMN extends ColumnType
case object FLOAT_COLUMN extends ColumnType
case object DOUBLE_COLUMN extends ColumnType

/**
 * A column is one basic property (primitive, String, etc.).
 */
class ColumnSchema(
    /** Type of the property. Is null when the whole object is a primitive. */
    val columnType: ColumnType,
    /** How many bytes does a single property take. */
    val bytes: Int,
    /** Scala terms with property name and other information */
    val terms: Vector[TermSymbol] = Vector[TermSymbol]()) {

  def name = terms.map(_.name)

}
