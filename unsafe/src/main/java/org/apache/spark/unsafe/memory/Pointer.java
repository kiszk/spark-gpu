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

import java.nio.Buffer;
import java.nio.ByteBuffer;


public final class Pointer {
  private jcuda.Pointer  jcudaPointer;
  public Pointer(jcuda.Pointer ptr) {
    jcudaPointer = ptr;
  }
  public jcuda.Pointer getJPointer() {
    return jcudaPointer;
  }
  public ByteBuffer getByteBuffer(long offset, long size) {
    return jcudaPointer.getByteBuffer(offset, size);
  }
  public static Pointer to(Buffer buffer) { return new Pointer(jcuda.Pointer.to(buffer)); }
  public static Pointer to(byte[] values) { return new Pointer(jcuda.Pointer.to(values)); }
  public static Pointer to(char[] values) { return new Pointer(jcuda.Pointer.to(values)); }
  public static Pointer to(double[] values) { return new Pointer(jcuda.Pointer.to(values)); }
  public static Pointer to(float[] values) { return new Pointer(jcuda.Pointer.to(values)); }
  public static Pointer to(int[] values) { return new Pointer(jcuda.Pointer.to(values)); }
  public static Pointer to(long[] values) { return new Pointer(jcuda.Pointer.to(values)); }
  public static Pointer to(short[] values) { return new Pointer(jcuda.Pointer.to(values)); }
  public static Pointer to(Pointer... pointers) {
    int len = pointers.length;
    jcuda.Pointer[] jpointers = new jcuda.Pointer[len];
    for (int i = 0; i < len; i++) { jpointers[i] = pointers[i].getJPointer(); }
    return new Pointer(jcuda.Pointer.to(jpointers));
  }
  public static Pointer to(jcuda.NativePointerObject... pointers) {
    return new Pointer(jcuda.Pointer.to(pointers));
  }
}
