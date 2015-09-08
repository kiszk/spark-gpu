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

import java.lang.ref.WeakReference;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Iterator;
import javax.annotation.concurrent.GuardedBy;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

/**
 * Manages memory for an executor. Individual operators / tasks allocate memory through
 * {@link TaskMemoryManager} objects, which obtain their memory from ExecutorMemoryManager.
 */
public class ExecutorMemoryManager {

  /**
   * Allocator, exposed for enabling untracked allocations of temporary data structures.
   */
  public final MemoryAllocator allocator;

  /**
   * Tracks whether memory will be allocated on the JVM heap or off-heap using sun.misc.Unsafe.
   */
  final boolean inHeap;

  /**
   * Maximum allocateable amount of pinned memory or negative if unlimited.
   */
  public final long maxPinnedMemory;

  @GuardedBy("this")
  private final Map<Long, LinkedList<WeakReference<MemoryBlock>>> bufferPoolsBySize =
    new HashMap<Long, LinkedList<WeakReference<MemoryBlock>>>();

  @GuardedBy("this")
  private long allocatedPinnedMemory = 0;

  // Pointer content won't release itself, so no need for WeakReference
  @GuardedBy("this")
  private final Map<Long, LinkedList<Pointer>> pinnedMemoryBySize =
    new HashMap<Long, LinkedList<Pointer>>();

  private final Map<Pointer, Long> pinnedMemorySizes = new HashMap<Pointer, Long>();

  private static final int POOLING_THRESHOLD_BYTES = 1024 * 1024;

  /**
   * Construct a new ExecutorMemoryManager.
   *
   * @param allocator the allocator that will be used
   */
  public ExecutorMemoryManager(MemoryAllocator allocator, long maxPinnedMemory) {
    this.inHeap = allocator instanceof HeapMemoryAllocator;
    this.allocator = allocator;
    this.maxPinnedMemory = maxPinnedMemory;
  }

  /**
   * Returns true if allocations of the given size should go through the pooling mechanism and
   * false otherwise.
   */
  private boolean shouldPool(long size) {
    // Very small allocations are less likely to benefit from pooling.
    // At some point, we should explore supporting pooling for off-heap memory, but for now we'll
    // ignore that case in the interest of simplicity.
    return size >= POOLING_THRESHOLD_BYTES && allocator instanceof HeapMemoryAllocator;
  }

  /**
   * Allocates a contiguous block of memory. Note that the allocated memory is not guaranteed
   * to be zeroed out (call `zero()` on the result if this is necessary).
   */
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
      return allocator.allocate(size);
    } else {
      return allocator.allocate(size);
    }
  }

  public void free(MemoryBlock memory) {
    final long size = memory.size();
    if (shouldPool(size)) {
      synchronized (this) {
        LinkedList<WeakReference<MemoryBlock>> pool = bufferPoolsBySize.get(size);
        if (pool == null) {
          pool = new LinkedList<WeakReference<MemoryBlock>>();
          bufferPoolsBySize.put(size, pool);
        }
        pool.add(new WeakReference<MemoryBlock>(memory));
      }
    } else {
      allocator.free(memory);
    }
  }

  /**
   * Allocates pinned memory suitable for CUDA and returns a wrapped native Pointer. Takes the
   * memory from the pool if available or tries to allocate if not.
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
   * Frees pinned memory pointer. In reality, it just returns it to the pool.
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
