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
  private[spark] def mirror: universe.Mirror =
    universe.runtimeMirror(Thread.currentThread().getContextClassLoader)

  var onlyLoadableClassesSupported: Boolean = false

  private[spark] def localTypeOf[T: TypeTag] = universe.typeTag[T].in(mirror).tpe

  def schemaFor[T: TypeTag]: ColumnPartitionSchema =
    schemaForType(localTypeOf[T])

  private[spark] def schemaForType(tpe: Type): ColumnPartitionSchema = {
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
      case t if !onlyLoadableClassesSupported || Utils.classIsLoadable(t.typeSymbol.asClass.fullName) => {
        val valVarMembers = t.erasure.members.view
          .filter(p => !p.isMethod && p.isTerm).map(_.asTerm)
          .filter(p => p.isVar || p.isVal)

        valVarMembers.foreach { p =>
          // TODO more checks
          // is final okay?
          if (p.isStatic) throw new UnsupportedOperationException(s"Column schema with static field ${p.fullName} not supported")
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

  def isPrimitive = columns.size == 1 && columns(0).name.isEmpty

  // TODO allow for dropping specific columns if some kind of optimizer detected that they are not needed
  def serialize(iter: Iterator[Any], columnBuffers: Seq[ByteBuffer]) {
    val mirror = ColumnPartitionSchema.mirror
    val getters = columns.map { col =>
      col.terms.foldLeft(identity[Any] _)((r, term) =>
          ((obj: Any) => mirror.reflect(obj).reflectField(term).get) compose r)
    }

    iter.foreach { obj =>
      for (((col, getter), buf) <- ((columns zip getters) zip columnBuffers)) {
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

  def deserialize(columnBuffers: Seq[ByteBuffer], count: Int): Iterator[Any] = {
    if (isPrimitive) {
      Iterator.continually {
        deserializeColumnValue(columns(0).columnType, columnBuffers(0))
      } take count
    } else {
      val mirror = ColumnPartitionSchema.mirror
      val setters = columns.map { col =>
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

        (obj: Any, value: Any) => mirror.reflect(get(obj)).reflectField(col.terms.last).set(value)
      }

      Iterator.continually {
        val obj = ClosureCleaner.instantiateClass(cls, null)

        for (((col, setter), buf) <- ((columns zip setters) zip columnBuffers)) {
          setter(obj, deserializeColumnValue(col.columnType, buf))
        }
        
        obj
      } take count
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

  def name = terms.map(_.name)

}
