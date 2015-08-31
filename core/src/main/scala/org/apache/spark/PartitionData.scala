package org.apache.spark

abstract class PartitionData[+T] {

  /**
   * A helper function for wrapping IteratedPartitionData's iterator, if this partition is of that
   * type */
  def wrapIterator[U >: T](f: (Iterator[T] => Iterator[U])): PartitionData[U] = this

  /**
   * Read the data as normal Java objects
   */
  def iterator: Iterator[T]

}
