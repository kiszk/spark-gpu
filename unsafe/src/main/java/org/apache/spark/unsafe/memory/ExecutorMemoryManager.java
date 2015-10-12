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

import jcuda.CUresult;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.CudaException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
  final long maxPinnedMemory;

  @GuardedBy("this")
  private final Map<Long, LinkedList<WeakReference<MemoryBlock>>> bufferPoolsBySize =
    new HashMap<Long, LinkedList<WeakReference<MemoryBlock>>>();

  // TODO Instead of allocating pinned memory allocate normal page-aligned memory (valloc with
  // JNI?) and pin it in CUDAManager. This way this page-aligned memory might be used on CPU too
  // without needless costly pinning.
  @GuardedBy("this")
  private long allocatedPinnedMemory = 0;

  // Pointer content won't release itself, so no need for WeakReference
  // TODO use ConcurrentHashMap instead of synchronization and maybe even do thread-local pooling
  // instead
  @GuardedBy("this")
  private final Map<Long, LinkedList<Pointer>> pinnedMemoryBySize =
    new HashMap<Long, LinkedList<Pointer>>();

  private final Map<Pointer, Long> pinnedMemorySizes = new HashMap<Pointer, Long>();

  private static final int POOLING_THRESHOLD_BYTES = 1024 * 1024;

  /**
   * Construct a new ExecutorMemoryManager with unlimited off-heap pinned memory.
   *
   * @param allocator the allocator that will be used
   */
  public ExecutorMemoryManager(MemoryAllocator allocator) {
    this(allocator, -1);
  }

  /**
   * Construct a new ExecutorMemoryManager.
   *
   * @param allocator the allocator that will be used
   * @param maxPinnedMemory the maximum amount of allocated off-heap pinned memory
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

      // Garbage collecting and deallocating some pinned memory, so that we have space for the new
      // one.
      // TODO might be better to start from LRU size, currently freeing in arbitrary order
      if (maxPinnedMemory >= 0 && allocatedPinnedMemory + size > maxPinnedMemory) {
        Iterator<Map.Entry<Long, LinkedList<Pointer>>> it =
          pinnedMemoryBySize.entrySet().iterator();

        while (it.hasNext() && allocatedPinnedMemory + size > maxPinnedMemory) {
          Map.Entry<Long, LinkedList<Pointer>> sizeAndList = it.next();
          assert(!sizeAndList.getValue().isEmpty());

          Iterator<Pointer> listIt = sizeAndList.getValue().iterator();

          do {
            Pointer ptr = listIt.next();
            try {
              int result = JCuda.cudaFreeHost(ptr);
              if (result != CUresult.CUDA_SUCCESS) {
                throw new CudaException(JCuda.cudaGetErrorString(result));
              }
            } catch (CudaException ex) {
              throw new OutOfMemoryError("Could not free pinned memory: " + ex.getMessage());
            }
            listIt.remove();
            pinnedMemorySizes.remove(ptr);
            allocatedPinnedMemory -= sizeAndList.getKey();
          } while (allocatedPinnedMemory + size < maxPinnedMemory && listIt.hasNext());

          if (!listIt.hasNext()) {
            it.remove();
          }
        }

        if (maxPinnedMemory >= 0 && allocatedPinnedMemory + size > maxPinnedMemory) {
          throw new OutOfMemoryError("Not enough free pinned memory");
        }
      }

      Pointer ptr = new Pointer();
      try {
        int result = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocPortable);
        if (result != CUresult.CUDA_SUCCESS) {
          throw new CudaException(JCuda.cudaGetErrorString(result));
        }
      } catch (CudaException ex) {
        throw new OutOfMemoryError("Could not alloc pinned memory: " + ex.getMessage());
      }
      pinnedMemorySizes.put(ptr, size);
      allocatedPinnedMemory += size;
      return ptr;
    }
  }

  /**
   * Frees off-heap pinned memory pointer. In reality, it just returns it to the pool.
   * Returns how much memory was freed.
   */
  public long freePinnedMemory(Pointer ptr) {
    synchronized (this) {
      final long size = pinnedMemorySizes.get(ptr);
      LinkedList<Pointer> pool = pinnedMemoryBySize.get(size);
      if (pool == null) {
        pool = new LinkedList<Pointer>();
        pinnedMemoryBySize.put(size, pool);
      }
      pool.add(ptr);
      return size;
    }
  }

  static protected final Logger logger = LoggerFactory.getLogger(ExecutorMemoryManager.class);

  protected void finalize() {
    // Deallocating off-heap pinned memory pool
    for (Map.Entry<Long, LinkedList<Pointer>> sizeAndList : pinnedMemoryBySize.entrySet()) {
      for (Pointer ptr : sizeAndList.getValue()) {
        try {
          int result = JCuda.cudaFreeHost(ptr);
          if (result != CUresult.CUDA_SUCCESS) {
            throw new CudaException(JCuda.cudaGetErrorString(result));
          }
          allocatedPinnedMemory -= sizeAndList.getKey();
        } catch (CudaException ex) {
          throw new OutOfMemoryError("Could not free pinned memory: " + ex.getMessage());
        }
      }
    }

    if (allocatedPinnedMemory > 0 && logger.isWarnEnabled()) {
      logger.warn("{}B of memory still not freed in finalizer.", allocatedPinnedMemory);
    }
  }

  /**
   * Returns amount of allocated pinned memory. For testing purposes.
   */
  long getAllocatedPinnedMemorySize() {
    synchronized (this) {
      return allocatedPinnedMemory;
    }
  }

  /**
   * Returns amount of allocated pinned memory that is not in the pool. For testing purposes.
   */
  long getUsedAllocatedPinnedMemorySize() {
    synchronized (this) {
      long size = allocatedPinnedMemory;

      for (Map.Entry<Long, LinkedList<Pointer>> sizeAndList : pinnedMemoryBySize.entrySet()) {
        size -= sizeAndList.getKey() * sizeAndList.getValue().size();
      }

      return size;
    }
  }

}
