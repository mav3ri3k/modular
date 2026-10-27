// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "qurt_thread.h"
extern unsigned qurt_thread_get_id(void);
typedef unsigned int u32;
typedef unsigned long long u64;
typedef union {
  struct {
    void *data;
    __SIZE_TYPE__ len;
  } buf;
  u32 handle;
  u64 handle64;
} rpc_arg;
extern int modular_dsp_hmx_tile(void *, void *, void *, void *);
extern void modular_dsp_hmx_clear(void);
extern void modular_dsp_affine(float *, float *, int);
static int power_context;
static void hmx_worker(void *data) {
  int *r = data;
  HAP_power_request_t p = {0};
  p.type = HAP_power_set_HMX_v2;
  p.hmx_v2.set_power = 1;
  p.hmx_v2.power_up = 1;
  r[0] = HAP_power_set(&power_context, &p);
  compute_res_attr_t attr;
  r[1] = HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_cache_mode(&attr, 1);
  r[2] = HAP_compute_res_attr_set_vtcm_param_v2(&attr, 32768, 32768, 32768);
  // v81 autonomous HMX allocation requires this legacy request to be OFF.
  r[3] = HAP_compute_res_attr_set_hmx_param(&attr, 0);
  unsigned tid = qurt_thread_get_id();
  r[14] = HAP_compute_res_attr_set_threads(&attr, &tid, 1);
  unsigned ctx = HAP_compute_res_acquire(&attr, 1000000);
  r[4] = ctx;
  if (ctx) {
    r[13] = HAP_compute_res_acquire_cached(ctx, 1000000);
    unsigned *busy = 0;
    r[5] = HAP_compute_res_hmx_lock4(ctx, &busy, 1000000);
    if (!r[5]) {
      void *vtcm = 0;
      unsigned size = 0;
      r[10] = HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm, &size);
      if (!r[10] && vtcm && size >= 8192) {
        *(volatile unsigned *)busy = 1;
        modular_dsp_hmx_clear();
        r[10] = modular_dsp_hmx_tile(vtcm, (char *)vtcm + 2048,
                                     (char *)vtcm + 4096, (char *)vtcm + 6144);
        *(volatile unsigned *)busy = 0;
        r[11] = *(unsigned short *)((char *)vtcm + 6144);
      }
      r[6] = HAP_compute_res_hmx_unlock4(ctx, busy);
    }
    compute_res_hmx_mutex_t mutex = {0};
    r[7] = HAP_compute_res_hmx_lock3(ctx, 0, &mutex, 1000000);
    if (!r[7])
      r[8] = HAP_compute_res_hmx_unlock3(ctx, 0, &mutex);
    HAP_compute_res_release_cached(ctx);
    r[9] = HAP_compute_res_release(ctx);
  }
  p.hmx_v2.power_up = 0;
  r[12] = HAP_power_set(&power_context, &p);
  qurt_thread_exit(0);
}
__attribute__((visibility("default"))) int
modular_hexagon_invoke(u64 handle, u32 scalars, rpc_arg *args) {
  if (!args)
    return 14;
  if (scalars == ((2u << 16) | 1u) && handle == 0) {
    args[2].handle64 = 1;
    return 0;
  }
  if (scalars == ((1u << 24) | (1u << 4)) && handle == 0)
    return 0;
  if (handle != 1)
    return 14;
  if (scalars == ((2u << 24) | (1u << 16) | (1u << 8))) {
    if (args[0].buf.len != args[1].buf.len || args[0].buf.len % 4)
      return 14;
    modular_dsp_affine(args[0].buf.data, args[1].buf.data, args[0].buf.len / 4);
    return 0;
  }
  if (scalars != ((5u << 24) | (1u << 8)) || args[0].buf.len != 16 * 4)
    return 14;
  int *r = args[0].buf.data;
  for (int i = 0; i < 16; ++i)
    r[i] = -1;
  static unsigned char stack[32768] __attribute__((aligned(128)));
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, sizeof(stack));
  qurt_thread_attr_set_priority(&attr, 128);
  qurt_thread_attr_set_name(&attr, "modular-hmx");
  unsigned tid;
  int err = qurt_thread_create(&tid, &attr, hmx_worker, r);
  if (err)
    r[15] = err;
  else {
    int status;
    qurt_thread_join(tid, &status);
    r[15] = status;
  }

  return 0;
}
