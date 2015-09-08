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

package org.apache.spark.unsafe.memory;

import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;

import javax.annotation.Nullable;

import org.apache.spark.unsafe.Platform;

/**
 * A consecutive block of memory, starting at a {@link MemoryLocation} with a fixed size.
 */
public class MemoryBlock extends MemoryLocation {

  private final long length;

  /**
   * Optional page number; used when this MemoryBlock represents a page allocated by a
   * TaskMemoryManager. This field is public so that it can be modified by the TaskMemoryManager,
   * which lives in a different package.
   */
  public int pageNumber = -1;

  public MemoryBlock(@Nullable Object obj, long offset, long length) {
    super(obj, offset);
    this.length = length;
  }

  /**
   * Returns the size of the memory block.
   */
  public long size() {
    return length;
  }

  /**
   * Returns a ByteBuffer backed by this memory. Experimental. Uses some rather private API and
   * reflection, so beware of issues when trying new JDKs and processors.
   * Even if memory block is larger, this does not support more than 2GB at once, because of
   * ByteBuffer limitations. Also, might be a source of problems when deallocating.
   */
  public ByteBuffer toByteBuffer() {
    if (length > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Can't make ByteBuffer from MemoryBlock larger than 2GB");
    }

    if (obj != null) {
      throw new IllegalArgumentException("On-heap memory not supported");
    }

    // Off-heap memory
    // Buffer class is public, though those members are private/package-private
    Field address, capacity;
    try {
      address = Buffer.class.getDeclaredField("address");
      address.setAccessible(true);
      capacity = Buffer.class.getDeclaredField("capacity");
      capacity.setAccessible(true);
    } catch (NoSuchFieldException ex) {
      throw new RuntimeException("Could not make Buffer class fields accessible", ex);
    }

    // This allocates an empty direct buffer, which should not have any off-heap memory allocated
    ByteBuffer buf = ByteBuffer.allocateDirect(0);

    try {
      address.setLong(buf, offset);
      capacity.setInt(buf, (int)length);
    } catch (IllegalAccessException ex) {
      throw new RuntimeException("Could not set Buffer class fields", ex);
    }

    return buf;
  }

  /**
   * Creates a memory block pointing to the memory used by the long array.
   */
  public static MemoryBlock fromLongArray(final long[] array) {
    return new MemoryBlock(array, Platform.LONG_ARRAY_OFFSET, array.length * 8);
  }
}
