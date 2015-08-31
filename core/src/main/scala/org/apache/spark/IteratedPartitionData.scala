package org.apache.spark

case class IteratedPartitionData[+T](
    iter: Iterator[T]
  ) extends PartitionData[T] {
    
  override def wrapIterator[U >: T](f: (Iterator[T] => Iterator[U])) =
    IteratedPartitionData(f(iter))

  def iterator = iter

}
