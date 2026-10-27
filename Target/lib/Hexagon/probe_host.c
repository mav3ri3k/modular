// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef union {
  struct {
    void *data;
    size_t len;
  } buf;
  uint32_t handle;
  uint64_t handle64;
} arg;
int main(void) {
  setvbuf(stdout, 0, _IOLBF, 0);
  void *lib = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW);
  if (!lib) {
    puts(dlerror());
    return 1;
  }
  int (*session)(uint32_t, void *, uint32_t) =
      dlsym(lib, "remote_session_control");
  int (*open)(const char *, uint64_t *) = dlsym(lib, "remote_handle64_open");
  int (*invoke)(uint64_t, uint32_t, arg *) =
      dlsym(lib, "remote_handle64_invoke");
  int (*close)(uint64_t) = dlsym(lib, "remote_handle64_close");
  struct {
    int domain, enable;
  } u = {3, 1};
  int rc = session(2, &u, sizeof(u));
  printf("unsigned=%d\n", rc);
  if (rc)
    return 2;
  uint64_t h = 0;
  rc =
      open("file:///"
           "libmodular_hexagon.so?modular_hexagon_invoke&_modver=1.0&_dom=cdsp",
           &h);
  printf("open=%d\n", rc);
  if (rc)
    return 3;
  float x[257], y[257];
  for (int i = 0; i < 257; ++i) {
    x[i] = i * .25f;
    y[i] = -1;
  }
  arg a[2] = {0};
  a[0].buf.data = x;
  a[0].buf.len = sizeof(x);
  a[1].buf.data = y;
  a[1].buf.len = sizeof(y);
  rc = invoke(h, (2u << 24) | (1u << 16) | (1u << 8), a);
  printf("Mojo DSP invoke=%d\n", rc);
  int fail = rc;
  for (int i = 0; i < 257; ++i)
    if (y[i] != 2 * x[i] + 1)
      fail = 1;
  printf("Mojo DSP affine: %s (257 outputs)\n", fail ? "FAIL" : "PASS");
  int r[16];
  arg b = {0};
  b.buf.data = r;
  b.buf.len = sizeof(r);
  rc = invoke(h, (5u << 24) | (1u << 8), &b);
  printf("HMX invoke=%d results:", rc);
  for (int i = 0; i < 16; ++i)
    printf(" %d", r[i]);
  puts("");
  if (rc || !r[4] || r[11] != 0x3c00)
    fail = 1;
  for (int i = 0; i < 16; ++i)
    if (i != 4 && i != 11 && r[i] != 0)
      fail = 1;
  printf(
      "Mojo HMX FP16 32x32x32 GEMM: %s (1024 outputs, nonuniform operands)\n",
      fail ? "FAIL" : "PASS");
  close(h);
  return fail;
}
