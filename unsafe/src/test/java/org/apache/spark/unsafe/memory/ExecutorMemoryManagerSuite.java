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

import jcuda.Pointer;

import org.junit.Assert;
import org.junit.Test;

public class ExecutorMemoryManagerSuite {

  @Test
  public void allocatedAndFreedMemoryIsPooled() {
    final ExecutorMemoryManager manager = new ExecutorMemoryManager(MemoryAllocator.HEAP, 4096);
    Assert.assertEquals(4096, manager.maxPinnedMemory);
    Assert.assertEquals(0, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedPinnedMemorySize());

    Pointer ptr1 = manager.allocatePinnedMemory(2048);
    Assert.assertEquals(2048, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(2048, manager.getUsedAllocatedPinnedMemorySize());

    Pointer ptr2 = manager.allocatePinnedMemory(1024);
    int ptr2AddressHash = ptr2.hashCode();
    Assert.assertEquals(3072, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(3072, manager.getUsedAllocatedPinnedMemorySize());

    manager.freePinnedMemory(ptr2);
    Assert.assertEquals(3072, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(2048, manager.getUsedAllocatedPinnedMemorySize());

    ptr2 = manager.allocatePinnedMemory(1024);
    Assert.assertEquals(ptr2AddressHash, ptr2.hashCode());
    Assert.assertEquals(3072, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(3072, manager.getUsedAllocatedPinnedMemorySize());

    manager.freePinnedMemory(ptr2);
    Assert.assertEquals(3072, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(2048, manager.getUsedAllocatedPinnedMemorySize());

    manager.freePinnedMemory(ptr1);
    Assert.assertEquals(3072, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedPinnedMemorySize());
  }

  @Test
  public void allocateAndFreeMaxMemoryWorks() {
    final ExecutorMemoryManager manager = new ExecutorMemoryManager(MemoryAllocator.HEAP, 4096);
    Assert.assertEquals(4096, manager.maxPinnedMemory);
    Assert.assertEquals(0, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedPinnedMemorySize());
    Pointer ptr = manager.allocatePinnedMemory(4096);
    Assert.assertEquals(4096, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(4096, manager.getUsedAllocatedPinnedMemorySize());
    manager.freePinnedMemory(ptr);
    Assert.assertEquals(4096, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedPinnedMemorySize());
  }

  @Test(expected=OutOfMemoryError.class)
  public void allocationOverMaxMemoryThrowsOOM() {
    final ExecutorMemoryManager manager = new ExecutorMemoryManager(MemoryAllocator.HEAP, 4096);
    Assert.assertEquals(4096, manager.maxPinnedMemory);
    Assert.assertEquals(0, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedPinnedMemorySize());
    manager.allocatePinnedMemory(8192);
  }

  @Test(expected=OutOfMemoryError.class)
  public void allocatingTooMuchThrowsOOM() {
    final ExecutorMemoryManager manager = new ExecutorMemoryManager(MemoryAllocator.HEAP, 4096);
    Assert.assertEquals(4096, manager.maxPinnedMemory);
    Assert.assertEquals(0, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedPinnedMemorySize());
    Pointer ptr = manager.allocatePinnedMemory(3072);
    Assert.assertEquals(3072, manager.getAllocatedPinnedMemorySize());
    Assert.assertEquals(3072, manager.getUsedAllocatedPinnedMemorySize());
    try {
      manager.allocatePinnedMemory(2048);
    } catch (Exception ex) {
      // This is to not leak the memory for real and to not have a warning logged
      manager.freePinnedMemory(ptr);
      throw ex;
    }
  }

}
