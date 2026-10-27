// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Exercise the actual C ABI used by the C++ HAL shim, with C-owned out slots.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
extern "C" {
const char *MojoHAL_DeviceContext_create(void **, const char *, int32_t);
void MojoHAL_DeviceContext_release(void *);
const char *MojoHAL_DeviceContext_createBuffer(void **, void **, void *,
                                               uint64_t);
void MojoHAL_DeviceBuffer_release(void *);
bool MojoHAL_DeviceBuffer_isOpaque(void *);
const char *MojoHAL_DeviceBuffer_getOpaqueView(uint64_t *, uint64_t *,
                                               uint64_t *, void *);
const char *MojoHAL_DeviceBuffer_createSubBuffer(void **, void *, uint64_t,
                                                 uint64_t);
const char *MojoHAL_DeviceBuffer_copyFromHost(void *, const void *, uint64_t);
const char *MojoHAL_DeviceBuffer_copyToHost(void *, void *, uint64_t, uint64_t);
const char *MojoHAL_DeviceBuffer_copy(void *, void *, uint64_t, uint64_t);
const char *MojoHAL_DeviceBuffer_fill(void *, uint64_t, uint64_t);
void MojoHAL_strfree(const char *);
void KGEN_CompilerRT_DestroyGlobals();
}
static void check(const char *error) {
  if (error) {
    std::fprintf(stderr, "FAIL: %s\n", error);
    MojoHAL_strfree(error);
    std::exit(1);
  }
}
#define EXPECT(expr)                                                           \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr);             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)
int main() {
  void *ctx = nullptr, *root = nullptr, *dest = nullptr, *sub = nullptr,
       *nested = nullptr;
  void *raw = reinterpret_cast<void *>(uintptr_t(1));
  check(MojoHAL_DeviceContext_create(&ctx, "opencl", 0));
  check(MojoHAL_DeviceContext_createBuffer(&root, &raw, ctx, 64));
  EXPECT(raw == nullptr && root && MojoHAL_DeviceBuffer_isOpaque(root));
  check(MojoHAL_DeviceContext_createBuffer(&dest, &raw, ctx, 64));
  float values[16], actual[16];
  for (int i = 0; i < 16; ++i)
    values[i] = float(i);
  check(MojoHAL_DeviceBuffer_copyFromHost(root, values, sizeof(values)));
  check(MojoHAL_DeviceBuffer_createSubBuffer(&sub, root, 4, 32));
  check(MojoHAL_DeviceBuffer_createSubBuffer(&nested, sub, 8, 16));
  uint64_t allocation = 0, offset = 0, size = 0;
  check(MojoHAL_DeviceBuffer_getOpaqueView(&allocation, &offset, &size, root));
  auto identity = allocation;
  check(
      MojoHAL_DeviceBuffer_getOpaqueView(&allocation, &offset, &size, nested));
  EXPECT(allocation == identity && offset == 12 && size == 16);
  MojoHAL_DeviceBuffer_release(sub);
  MojoHAL_DeviceBuffer_release(root);
  check(MojoHAL_DeviceBuffer_copyToHost(actual, nested, 0, 16));
  for (int i = 0; i < 4; ++i)
    EXPECT(actual[i] == float(i + 3));
  check(MojoHAL_DeviceBuffer_fill(nested, 0x40000000, 4));
  check(MojoHAL_DeviceBuffer_fill(dest, 0, 1));
  check(MojoHAL_DeviceBuffer_copy(dest, nested, 0, 16));
  check(MojoHAL_DeviceBuffer_copyToHost(actual, dest, 0, sizeof(actual)));
  for (int i = 0; i < 16; ++i)
    EXPECT(actual[i] == (i < 4 ? 2.0f : 0.0f));
  void *invalid = nullptr;
  auto *error = MojoHAL_DeviceBuffer_createSubBuffer(&invalid, nested, 16, 1);
  EXPECT(error != nullptr);
  MojoHAL_strfree(error);
  MojoHAL_DeviceBuffer_release(nested);
  MojoHAL_DeviceBuffer_release(dest);
  MojoHAL_DeviceContext_release(ctx);
  KGEN_CompilerRT_DestroyGlobals();
  std::puts("PASS: C ABI opaque allocation, nested views, typed copies, fills "
            "and parent lifetime");
}
