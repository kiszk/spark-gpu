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
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Iterator;

import org.apache.spark.unsafe.Platform;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * A simple {@link MemoryAllocator} that can allocate up to 16GB using a JVM long primitive array.
 */
public class HeapMemoryAllocator implements MemoryAllocator {

  /**
   * Maximum allocateable amount of off-heap memory or negative if unlimited.
   */
  final long maxMemory;

  final int  memType;	// 0:default, 1:pinned, 2:array

  final boolean isMemPool; 

  @GuardedBy("this")
  private final Map<Long, LinkedList<WeakReference<MemoryBlock>>> bufferPoolsBySize =
          new HashMap<>();

  // TODO instead of allocating memory, allocate normal page-aligned memory (valloc with
  // JNI?) and pin it in CUDAManager
  // This way this page-aligned memory might be used on CPU too without needless costly pinning
  @GuardedBy("this")
  private long allocatedMemory = 0;

  // Pointer content won't release itself, so no need for WeakReference
  @GuardedBy("this")
  private final Map<Long, LinkedList<Pointer>> memoryBySize =
          new HashMap<Long, LinkedList<Pointer>>();

  private final Map<Pointer, Long> memorySizes = new HashMap<Pointer, Long>();

  private static final int POOLING_THRESHOLD_BYTES = 1024 * 1024;

  /**
   * Construct a new HeapMemoryAllocator with unlimited off-heap memory.
   */
  public HeapMemoryAllocator() {
    this(-1, "default", true);
  }

  public HeapMemoryAllocator(long maxMemory) {
      this(maxMemory, "default", true);
  }

  /**
   * Construct a new HeapMemoryAllocator.
   *
   * @param maxMemory the maximum amount of allocated off-heap memory
   */
  public HeapMemoryAllocator(long maxMemory, String memType, boolean isMemPool) {
    this.maxMemory = maxMemory;
    this.isMemPool = isMemPool;
    if ("default".equals(memType)) {
      this.memType = 0;
    } else if ("pinned".equals(memType)) {
      this.memType = 1;
    } else if ("array".equals(memType)) {
      this.memType = 2;
    } else {
      throw new jcuda.CudaException("No support of memType:" + memType);
    }
  }

  /**
   * Returns true if allocations of the given size should go through the pooling mechanism and
   * false otherwise.
   */
  private boolean shouldPool(long size) {
    // Very small allocations are less likely to benefit from pooling.
    return isMemPool && (size >= POOLING_THRESHOLD_BYTES);
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
   * Allocates off-heap memory suitable for CUDA and returns a wrapped native Pointer. Takes
   * the memory from the pool if available or tries to allocate if not.
   */
  public Pointer allocateMemory(long size) {
    if (maxMemory >= 0 && size > maxMemory) {
      throw new OutOfMemoryError("Trying to allocate more memory than the total limit");
    }

    synchronized (this) {
      final LinkedList<Pointer> pool = memoryBySize.get(size);
      if (pool != null) {
        assert (!pool.isEmpty());
        final Pointer ptr = pool.pop();
        if (pool.isEmpty()) {
          memoryBySize.remove(size);
        }
        return ptr;
      }

      // Garbage collecting and deallocating some memory, so that we have space for the new
      // one.
      // TODO might be better to start from LRU size, currently freeing in arbitrary order
      if (maxMemory >= 0 && allocatedMemory + size > maxMemory) {
        Iterator<Map.Entry<Long, LinkedList<Pointer>>> it =
                memoryBySize.entrySet().iterator();

        while (it.hasNext() && allocatedMemory + size > maxMemory) {
          Map.Entry<Long, LinkedList<Pointer>> sizeAndList = it.next();
          assert (!sizeAndList.getValue().isEmpty());

          Iterator<Pointer> listIt = sizeAndList.getValue().iterator();

          do {
            Pointer ptr = listIt.next();
            _freeMemory(ptr);
            listIt.remove();
            memorySizes.remove(ptr);
            allocatedMemory -= sizeAndList.getKey();
          } while (allocatedMemory + size < maxMemory && listIt.hasNext());

          if (!listIt.hasNext()) {
            it.remove();
          }
        }

        if (maxMemory >= 0 && allocatedMemory + size > maxMemory) {
          throw new OutOfMemoryError("Not enough free memory");
        }
      }

      Pointer ptr = _allocateMemory(size);
      memorySizes.put(ptr, size);
      allocatedMemory += size;
      return ptr;
    }
  }

  /**
   * Frees off-heap memory pointer. In reality, it just returns it to the pool.
   * Returns how much memory was freed.
   */
  public long freeMemory(Pointer ptr) {
    synchronized (this) {
      final long size = memorySizes.get(ptr);
      LinkedList<Pointer> pool = memoryBySize.get(size);
      if (pool == null) {
        pool = new LinkedList<Pointer>();
        memoryBySize.put(size, pool);
      }
      pool.add(ptr);
      return size;
    }
  }

  static protected final Logger logger = LoggerFactory.getLogger(HeapMemoryAllocator.class);

  protected void finalize() {
    // Deallocating off-heap memory pool
    for (Map.Entry<Long, LinkedList<Pointer>> sizeAndList : memoryBySize.entrySet()) {
      for (Pointer ptr : sizeAndList.getValue()) {
        _freeMemory(ptr);
        allocatedMemory -= sizeAndList.getKey();
      }
    }

    if (allocatedMemory > 0 && logger.isWarnEnabled()) {
      logger.warn("{}B of memory still not freed in finalizer.", allocatedMemory);
    }
  }

  /**
   * Returns amount of allocated memory. For testing purposes.
   */
  long getAllocatedMemorySize() {
    synchronized (this) {
      return allocatedMemory;
    }
  }

  /**
   * Returns amount of allocated memory that is not in the pool. For testing purposes.
   */
  long getUsedAllocatedMemorySize() {
    synchronized (this) {
      long size = allocatedMemory;

      for (Map.Entry<Long, LinkedList<Pointer>> sizeAndList : memoryBySize.entrySet()) {
        size -= sizeAndList.getKey() * sizeAndList.getValue().size();
      }

      return size;
    }
  }


  private Pointer _allocateMemory(long size) {
    jcuda.Pointer ptr = new jcuda.Pointer();
    if ((memType == 0) || (memType == 1)) {
      try {
        int result = jcuda.runtime.JCuda.cudaHostAlloc(ptr, size,
	  (memType == 0) ? jcuda.runtime.JCuda.cudaHostAllocDefault :
                           jcuda.runtime.JCuda.cudaHostAllocPortable);
        if (result != jcuda.driver.CUresult.CUDA_SUCCESS) {
          throw new jcuda.CudaException(jcuda.runtime.JCuda.cudaGetErrorString(result));
        }
      } catch (jcuda.CudaException ex) {
        throw new OutOfMemoryError("Could not alloc memory: " + ex.getMessage());
      }
    } else if (memType == 2) {
      if ((long)Integer.MAX_VALUE < size) {
        throw new jcuda.CudaException("allocation size ("+size+") must be less than 2^31");
      }
      ptr.to(new byte[(int)size]);
    }
    return new Pointer(ptr);
  }

  private void _freeMemory(Pointer ptr) {
    if ((memType == 0) || (memType == 1)) {
      try {
        int result = jcuda.runtime.JCuda.cudaFreeHost(ptr.getJPointer());
        if (result != jcuda.driver.CUresult.CUDA_SUCCESS) {
          throw new jcuda.CudaException(jcuda.runtime.JCuda.cudaGetErrorString(result));
        }
      } catch (jcuda.CudaException ex) {
        throw new OutOfMemoryError("Could not free memory: " + ex.getMessage());
      }
    }
  }
}
