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

import javax.annotation.concurrent.GuardedBy;
import java.lang.ref.WeakReference;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import org.apache.spark.unsafe.Platform;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

/**
 * A simple {@link MemoryAllocator} that can allocate up to 16GB using a JVM long primitive array.
 */
public class HeapMemoryAllocator implements MemoryAllocator {

  /**
   * Maximum allocateable amount of pinned memory or negative if unlimited.
   */
  public final long maxPinnedMemory;

  @GuardedBy("this")
  private final Map<Long, LinkedList<WeakReference<MemoryBlock>>> bufferPoolsBySize =
    new HashMap<>();

  @GuardedBy("this")
  private long allocatedPinnedMemory = 0;

  // Pointer content won't release itself, so no need for WeakReference
  @GuardedBy("this")
  private final Map<Long, LinkedList<Pointer>> pinnedMemoryBySize =
    new HashMap<Long, LinkedList<Pointer>>();

  private final Map<Pointer, Long> pinnedMemorySizes = new HashMap<Pointer, Long>();

  private static final int POOLING_THRESHOLD_BYTES = 1024 * 1024;

  /**
   * Returns true if allocations of the given size should go through the pooling mechanism and
   * false otherwise.
   */
  private boolean shouldPool(long size) {
    // Very small allocations are less likely to benefit from pooling.
    return size >= POOLING_THRESHOLD_BYTES;
  }

  @Override
  public MemoryBlock allocate(long size) throws OutOfMemoryError {
    if (shouldPool(size)) {
      synchronized (this) {
        final LinkedList<WeakReference<MemoryBlock>> pool = bufferPoolsBySize.get(size);
        if (pool != null) {
          while (!pool.isEmpty()) {
            final WeakReference<MemoryBlock> blockReference = pool.pop();
            final MemoryBlock memory = blockReference.get();
            if (memory != null) {
              assert (memory.size() == size);
              return memory;
            }
          }
          bufferPoolsBySize.remove(size);
        }
      }
    }
    long[] array = new long[(int) ((size + 7) / 8)];
    return new MemoryBlock(array, Platform.LONG_ARRAY_OFFSET, size);
  }

  @Override
  public void free(MemoryBlock memory) {
    final long size = memory.size();
    if (shouldPool(size)) {
      synchronized (this) {
        LinkedList<WeakReference<MemoryBlock>> pool = bufferPoolsBySize.get(size);
        if (pool == null) {
          pool = new LinkedList<>();
          bufferPoolsBySize.put(size, pool);
        }
        pool.add(new WeakReference<>(memory));
      }
    } else {
      // Do nothing
    }
  }

  /**
   * Allocates off-heap pinned memory suitable for CUDA and returns a wrapped native Pointer. Takes
   * the memory from the pool if available or tries to allocate if not.
   */
  public Pointer allocatePinnedMemory(long size) {
    if (maxPinnedMemory >= 0 && size > maxPinnedMemory) {
      throw new OutOfMemoryError("Trying to allocate more pinned memory than the total limit");
    }

    synchronized (this) {
      final LinkedList<Pointer> pool = pinnedMemoryBySize.get(size);
      if (pool != null) {
        assert(!pool.isEmpty());
        final Pointer ptr = pool.pop();
        if (pool.isEmpty()) {
          pinnedMemoryBySize.remove(size);
        }
        return ptr;
      }

      // Collecting and deallocating some pinned memory, so that we have space for the new one.
      // TODO might be better to start from LRU size, currently freeing in arbitrary order
      if (maxPinnedMemory >= 0 && allocatedPinnedMemory + size < maxPinnedMemory) {
        Iterator<Map.Entry<Long, LinkedList<Pointer>>> it =
          pinnedMemoryBySize.entrySet().iterator();
        while (allocatedPinnedMemory + size < maxPinnedMemory) {
          assert(it.hasNext());
          Map.Entry<Long, LinkedList<Pointer>> sizeAndList = it.next();
          assert(!sizeAndList.getValue().isEmpty());

          Iterator<Pointer> listIt = sizeAndList.getValue().iterator();

          do {
            Pointer ptr = listIt.next();
            int error = JCuda.cudaFreeHost(ptr);
            if (error != 0) {
              throw new OutOfMemoryError("Could not free pinned memory (CUDA error " + error + ")");
            }
            listIt.remove();
            pinnedMemorySizes.remove(ptr);
            allocatedPinnedMemory -= sizeAndList.getKey();
          } while (allocatedPinnedMemory + size < maxPinnedMemory && listIt.hasNext());

          if (!listIt.hasNext()) {
            it.remove();
          }
        }
      }
    }

    Pointer ptr = new Pointer();
    int error = JCuda.cudaMallocHost(ptr, size);
    if (error != 0) {
      throw new OutOfMemoryError("Could not allocate pinned memory (CUDA error " + error + ")");
    }
    pinnedMemorySizes.put(ptr, size);
    allocatedPinnedMemory += size;
    return ptr;
  }

  /**
   * Frees off-heap pinned memory pointer. In reality, it just returns it to the pool.
   */
  public void freePinnedMemory(Pointer ptr) {
    synchronized (this) {
      final long size = pinnedMemorySizes.get(ptr);
      LinkedList<Pointer> pool = pinnedMemoryBySize.get(size);
      if (pool == null) {
        pool = new LinkedList<Pointer>();
        pinnedMemoryBySize.put(size, pool);
      }
      pool.add(ptr);
    }
  }
}
