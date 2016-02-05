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

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class HeapMemoryAllocatorSuite {

  @BeforeClass
  public static void setUp() {
    // normally it's set by CUDAManager
    jcuda.driver.JCudaDriver.setExceptionsEnabled(true);
    jcuda.driver.JCudaDriver.cuInit(0);
    jcuda.runtime.JCuda.cudaSetDevice(0);
  }

  @Test
  public void allocatedAndFreedMemoryIsPooled() {
    final HeapMemoryAllocator manager = new HeapMemoryAllocator(4096);
    Assert.assertEquals(4096, manager.maxMemory);
    Assert.assertEquals(0, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());

    Pointer ptr1 = manager.allocateMemory(2048);
    // this is a way of checking that ptr1 is not null internally (native address is private)
    //Assert.assertTrue(!ptr1.equals(new Pointer()));
    Assert.assertEquals(2048, manager.getAllocatedMemorySize());
    Assert.assertEquals(2048, manager.getUsedAllocatedMemorySize());

    Pointer ptr2 = manager.allocateMemory(1024);
    Assert.assertTrue(!ptr1.equals(ptr2));
    Assert.assertEquals(3072, manager.getAllocatedMemorySize());
    Assert.assertEquals(3072, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(1024, manager.freeMemory(ptr2));
    Assert.assertEquals(3072, manager.getAllocatedMemorySize());
    Assert.assertEquals(2048, manager.getUsedAllocatedMemorySize());

    Pointer ptr3 = manager.allocateMemory(1024);
    Assert.assertTrue(ptr2 == ptr3);
    Assert.assertTrue(ptr1 != ptr3);
    Assert.assertTrue(!ptr1.equals(ptr3));
    Assert.assertEquals(3072, manager.getAllocatedMemorySize());
    Assert.assertEquals(3072, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(manager.freeMemory(ptr3), 1024);
    Assert.assertEquals(3072, manager.getAllocatedMemorySize());
    Assert.assertEquals(2048, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(2048, manager.freeMemory(ptr1));
    Assert.assertEquals(3072, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());
  }

  @Test
  public void allocateAndFreeMaxMemoryWorks() {
    final HeapMemoryAllocator manager = new HeapMemoryAllocator(4096);
    Assert.assertEquals(4096, manager.maxMemory);
    Assert.assertEquals(0, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());

    Pointer ptr = manager.allocateMemory(4096);
    Assert.assertEquals(4096, manager.getAllocatedMemorySize());
    Assert.assertEquals(4096, manager.getUsedAllocatedMemorySize());

    manager.freeMemory(ptr);
    Assert.assertEquals(4096, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());
  }

  @Test
  public void gcToAllocateMemoryWorks() {
    final HeapMemoryAllocator manager = new HeapMemoryAllocator(4096);
    Assert.assertEquals(4096, manager.maxMemory);
    Assert.assertEquals(0, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());

    Pointer ptr1 = manager.allocateMemory(1024);
    Assert.assertEquals(1024, manager.getAllocatedMemorySize());
    Assert.assertEquals(1024, manager.getUsedAllocatedMemorySize());

    manager.freeMemory(ptr1);
    Assert.assertEquals(1024, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());

    Pointer ptr2 = manager.allocateMemory(4096);
    Assert.assertEquals(4096, manager.getAllocatedMemorySize());
    Assert.assertEquals(4096, manager.getUsedAllocatedMemorySize());

    manager.freeMemory(ptr2);
    Assert.assertEquals(4096, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());
  }

  @Test(expected=OutOfMemoryError.class)
  public void allocationOverMaxMemoryThrowsOOM() {
    final HeapMemoryAllocator manager = new HeapMemoryAllocator(4096);
    Assert.assertEquals(4096, manager.maxMemory);
    Assert.assertEquals(0, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());
    manager.allocateMemory(8192);
  }

  @Test(expected=OutOfMemoryError.class)
  public void allocatingTooMuchThrowsOOM() {
    final HeapMemoryAllocator manager = new HeapMemoryAllocator(4096);
    Assert.assertEquals(4096, manager.maxMemory);
    Assert.assertEquals(0, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());

    Pointer ptr = manager.allocateMemory(3072);
    Assert.assertEquals(3072, manager.getAllocatedMemorySize());
    Assert.assertEquals(3072, manager.getUsedAllocatedMemorySize());

    try {
      manager.allocateMemory(2048);
    } catch (Exception ex) {
      // This is to not leak the memory for real and to not have a warning logged
      manager.freeMemory(ptr);
      throw ex;
    }
  }

  @Test
  public void registersAllocatedSizesCorrectly() {
    final HeapMemoryAllocator manager = new HeapMemoryAllocator(4096);
    Pointer ptr1 = manager.allocateMemory(1);
    Assert.assertEquals(1, manager.getAllocatedMemorySize());
    Assert.assertEquals(1, manager.getUsedAllocatedMemorySize());

    Pointer ptr2 = manager.allocateMemory(1);
    Assert.assertEquals(2, manager.getAllocatedMemorySize());
    Assert.assertEquals(2, manager.getUsedAllocatedMemorySize());

    Pointer ptr3 = manager.allocateMemory(2);
    Assert.assertEquals(4, manager.getAllocatedMemorySize());
    Assert.assertEquals(4, manager.getUsedAllocatedMemorySize());

    Pointer ptr4 = manager.allocateMemory(1);
    Assert.assertEquals(5, manager.getAllocatedMemorySize());
    Assert.assertEquals(5, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(1, manager.freeMemory(ptr2));
    Assert.assertEquals(5, manager.getAllocatedMemorySize());
    Assert.assertEquals(4, manager.getUsedAllocatedMemorySize());

    Pointer ptr5 = manager.allocateMemory(4);
    Assert.assertEquals(9, manager.getAllocatedMemorySize());
    Assert.assertEquals(8, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(4, manager.freeMemory(ptr5));
    Assert.assertEquals(9, manager.getAllocatedMemorySize());
    Assert.assertEquals(4, manager.getUsedAllocatedMemorySize());

    Pointer ptr6 = manager.allocateMemory(8);
    Assert.assertEquals(17, manager.getAllocatedMemorySize());
    Assert.assertEquals(12, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(1, manager.freeMemory(ptr4));
    Assert.assertEquals(17, manager.getAllocatedMemorySize());
    Assert.assertEquals(11, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(2, manager.freeMemory(ptr3));
    Assert.assertEquals(17, manager.getAllocatedMemorySize());
    Assert.assertEquals(9, manager.getUsedAllocatedMemorySize());

    Pointer ptr7 = manager.allocateMemory(2);
    Assert.assertEquals(17, manager.getAllocatedMemorySize());
    Assert.assertEquals(11, manager.getUsedAllocatedMemorySize());

    Pointer ptr8 = manager.allocateMemory(8);
    Assert.assertEquals(25, manager.getAllocatedMemorySize());
    Assert.assertEquals(19, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(1, manager.freeMemory(ptr1));
    Assert.assertEquals(25, manager.getAllocatedMemorySize());
    Assert.assertEquals(18, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(8, manager.freeMemory(ptr6));
    Assert.assertEquals(25, manager.getAllocatedMemorySize());
    Assert.assertEquals(10, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(8, manager.freeMemory(ptr8));
    Assert.assertEquals(25, manager.getAllocatedMemorySize());
    Assert.assertEquals(2, manager.getUsedAllocatedMemorySize());

    Assert.assertEquals(2, manager.freeMemory(ptr7));
    Assert.assertEquals(25, manager.getAllocatedMemorySize());
    Assert.assertEquals(0, manager.getUsedAllocatedMemorySize());
  }

}
