// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Persistent single-session FastRPC projector. All projection math is Mojo.
// The SDK qurt_hvx header requires size_t to be defined first.
#include <stddef.h>

#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "qurt_anysignal.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"
#include <stdint.h>

typedef union {
  struct {
    void *data;
    __SIZE_TYPE__ len;
  } buf;
  unsigned handle;
  unsigned long long handle64;
} rpc_arg;
extern void hexagon_prepare_activations(void *, void *, int);
extern void hexagon_prepare_weights(void *, void *, int, int, int, int);
extern void hexagon_hmx_project(void *, void *, void *, void *, int, int);
extern void hexagon_reduce_projection(void *, void *, void *, void *);
#ifndef MODULAR_HEXAGON_HVX_WORKERS
#define MODULAR_HEXAGON_HVX_WORKERS 4
#endif
#define HELPERS (MODULAR_HEXAGON_HVX_WORKERS - 1)
static struct helper {
  unsigned tid, first, count, offset, phase;
  int status;
  qurt_anysignal_t request, complete;
  unsigned char stack[16384] __attribute__((aligned(128)));
} helpers[MODULAR_HEXAGON_HVX_WORKERS];
static struct {
  unsigned width, k, ctx, producer, matrix;
  const void *w, *scales;
  const void *x;
  void *y, *vtcm, *a, *tiles, *partials, *unit;
  int status, matrix_status;
  unsigned count, batch;
  unsigned debug[16];
  qurt_anysignal_t request, complete, matrix_request, matrix_complete;
  unsigned char stack[32768] __attribute__((aligned(128)));
  unsigned char matrix_stack[16384] __attribute__((aligned(128)));
} s;
static int power(int enabled) {
  HAP_power_request_t p = {0};
  p.type = HAP_power_set_HMX_v2;
  p.hmx_v2.set_power = 1;
  p.hmx_v2.power_up = enabled;
  p.hmx_v2.set_clock = enabled;
  p.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_TUR;
  p.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
  int rc = HAP_power_set(&s, &p);
  if (rc)
    return rc;
  p = (HAP_power_request_t){0};
  p.type = HAP_power_set_HVX;
  p.hvx.power_up = enabled;
  rc = HAP_power_set(&s, &p);
  if (rc || !enabled)
    return rc;
  p = (HAP_power_request_t){0};
  p.type = HAP_power_set_DCVS_v3;
  p.dcvs_v3.set_dcvs_enable = 1;
  p.dcvs_v3.dcvs_enable = 0;
  p.dcvs_v3.set_core_params = 1;
  p.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO;
  p.dcvs_v3.set_bus_params = 1;
  p.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO;
  return HAP_power_set(&s, &p);
}
static unsigned long long ticks(void) {
  unsigned long long t;
  __asm__ volatile("%0 = c31:30" : "=r"(t));
  return t;
}
static void matrix_worker(void *unused) {
  (void)unused;
  unsigned *busy = 0;
  for (;;) {
    unsigned command = qurt_anysignal_wait(&s.matrix_request, 7);
    qurt_anysignal_clear(&s.matrix_request, command);
    if (command & 2)
      break;
    unsigned long long start = ticks();
    if (command & 4) {
      s.matrix_status = busy ? HAP_compute_res_hmx_unlock4(s.ctx, busy) : 0;
      busy = 0;
    } else {
      s.matrix_status =
          busy ? 0 : HAP_compute_res_hmx_lock4(s.ctx, &busy, 1000000);
      if (!s.matrix_status) {
        *(volatile unsigned *)busy = 1;
        hexagon_hmx_project(s.a, s.tiles, s.unit, s.partials, s.k, s.count);
        *(volatile unsigned *)busy = 0;
      }
    }
    s.debug[1] += (unsigned)(ticks() - start);
    qurt_anysignal_set(&s.matrix_complete, 1);
  }
  if (busy)
    HAP_compute_res_hmx_unlock4(s.ctx, busy);
  qurt_thread_exit(0);
}
static void run_chunk(unsigned first, unsigned count, unsigned offset,
                      unsigned phase) {
  if (!count)
    return;
  if (phase == 1)
    hexagon_prepare_weights((void *)s.w, (char *)s.tiles + offset * s.k * 2,
                            s.width, s.k, first, count);
  else {
    int dims[4] = {s.width, s.k, first, count};
    hexagon_reduce_projection((char *)s.partials +
                                  (offset / 32) * (s.k / 128) * 2048,
                              (void *)s.scales, s.y, dims);
  }
}
static void helper_worker(void *data) {
  struct helper *h = data;
  h->status = qurt_hvx_lock(QURT_HVX_MODE_128B);
  qurt_anysignal_set(&h->complete, 1);
  if (!h->status) {
    for (;;) {
      unsigned command = qurt_anysignal_wait(&h->request, 3);
      qurt_anysignal_clear(&h->request, command);
      if (command & 2)
        break;
      run_chunk(h->first, h->count, h->offset, h->phase);
      qurt_anysignal_set(&h->complete, 1);
    }
    qurt_hvx_unlock();
  }
  qurt_thread_exit(0);
}
static void parallel_chunk(unsigned first, unsigned count, unsigned phase) {
  unsigned chunk = ((count / 32 + MODULAR_HEXAGON_HVX_WORKERS - 1) /
                    MODULAR_HEXAGON_HVX_WORKERS) *
                   32;
  for (unsigned i = 0; i < HELPERS; ++i) {
    unsigned offset = (i + 1) * chunk;
    helpers[i].first = first + offset;
    helpers[i].offset = offset;
    helpers[i].phase = phase;
    helpers[i].count =
        offset >= count ? 0 : (count - offset < chunk ? count - offset : chunk);
    qurt_anysignal_set(&helpers[i].request, 1);
  }
  run_chunk(first, count < chunk ? count : chunk, 0, phase);
  for (unsigned i = 0; i < HELPERS; ++i) {
    qurt_anysignal_wait(&helpers[i].complete, 1);
    qurt_anysignal_clear(&helpers[i].complete, 1);
  }
}
static void producer_worker(void *unused) {
  (void)unused;
  s.status = power(1);
  compute_res_attr_t attr;
  unsigned bytes = 0;
  if (!s.status)
    s.status = HAP_compute_res_attr_init(&attr);
  if (!s.status)
    s.status = HAP_compute_res_attr_set_cache_mode(&attr, 1);
  if (!s.status)
    s.status = HAP_compute_res_attr_set_hmx_param(&attr, 0);
  // 256 outputs: FP16 weight tiles + per-group partial tiles + activation
  // tiles.
  s.batch = s.k <= 2048 ? 1024 : 256;
  unsigned requested =
      s.k * 64 + s.k * s.batch * 2 + (s.k / 128) * (s.batch / 32) * 2048 + 4096;
  requested = (requested + 65535) & ~65535u;
  if (!s.status)
    s.status = HAP_compute_res_attr_set_vtcm_param_v2(&attr, requested,
                                                      requested, requested);
  unsigned tids[MODULAR_HEXAGON_HVX_WORKERS + 1] = {qurt_thread_get_id(),
                                                    s.matrix};
  for (unsigned i = 0; i < HELPERS; ++i)
    tids[i + 2] = helpers[i].tid;
  if (!s.status)
    s.status = HAP_compute_res_attr_set_threads(
        &attr, tids, MODULAR_HEXAGON_HVX_WORKERS + 1);
  if (!s.status) {
    s.ctx = HAP_compute_res_acquire(&attr, 1000000);
    if (!s.ctx)
      s.status = 1;
  }
  if (!s.status)
    s.status = HAP_compute_res_acquire_cached(s.ctx, 1000000);
  if (!s.status)
    s.status = HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &s.vtcm, &bytes);
  if (!s.status && bytes < requested)
    s.status = 1;
  if (!s.status) {
    s.a = s.vtcm;
    s.tiles = (char *)s.a + s.k * 64;
    s.partials = (char *)s.tiles + s.k * s.batch * 2;
    s.unit = (char *)s.partials + (s.k / 128) * (s.batch / 32) * 2048;
    unsigned *unit = s.unit;
    for (unsigned i = 0; i < 64; ++i)
      unit[i] = i < 32 ? 0x3c00 : 0;
    s.status = HAP_compute_res_release_cached(s.ctx);
  }
  qurt_anysignal_set(&s.complete, 1);
  if (!s.status)
    for (;;) {
      unsigned command = qurt_anysignal_wait(&s.request, 3);
      qurt_anysignal_clear(&s.request, command);
      if (command & 2)
        break;
      s.status = HAP_compute_res_acquire_cached(s.ctx, 1000000);
      int hvx_locked = 0;
      if (!s.status) {
        s.status = qurt_hvx_lock(QURT_HVX_MODE_128B);
        hvx_locked = !s.status;
      }
      for (unsigned i = 0; i < 16; ++i)
        s.debug[i] = 0;
      unsigned long long start = ticks();
      if (!s.status) {
        unsigned *unit = s.unit;
        for (unsigned i = 0; i < 64; ++i)
          unit[i] = i < 32 ? 0x3c00 : 0;
      }
      if (!s.status)
        hexagon_prepare_activations((void *)s.x, s.a, s.k);
      s.debug[4] = (unsigned)(ticks() - start);
      for (unsigned first = 0; !s.status && first < s.width; first += s.batch) {
        s.count = s.width - first < s.batch ? s.width - first : s.batch;
        start = ticks();
        parallel_chunk(first, s.count, 1);
        s.debug[0] += (unsigned)(ticks() - start);
        start = ticks();
        qurt_anysignal_set(&s.matrix_request, 1);
        qurt_anysignal_wait(&s.matrix_complete, 1);
        qurt_anysignal_clear(&s.matrix_complete, 1);
        s.debug[3] += (unsigned)(ticks() - start);
        s.status = s.matrix_status;
        start = ticks();
        if (!s.status)
          parallel_chunk(first, s.count, 2);
        s.debug[2] += (unsigned)(ticks() - start);
      }
      qurt_anysignal_set(&s.matrix_request, 4);
      qurt_anysignal_wait(&s.matrix_complete, 1);
      qurt_anysignal_clear(&s.matrix_complete, 1);
      if (!s.status)
        s.status = s.matrix_status;
      if (hvx_locked)
        qurt_hvx_unlock();
      int release_status = HAP_compute_res_release_cached(s.ctx);
      if (!s.status)
        s.status = release_status;
      qurt_anysignal_set(&s.complete, 1);
    }
  if (s.ctx)
    HAP_compute_res_release(s.ctx);
  power(0);
  HAP_power_destroy(&s);
  qurt_thread_exit(0);
}
static int start_thread(unsigned *tid, void (*fn)(void *), void *stack,
                        unsigned size, const char *name) {
  qurt_thread_attr_t a;
  qurt_thread_attr_init(&a);
  qurt_thread_attr_set_stack_addr(&a, stack);
  qurt_thread_attr_set_stack_size(&a, size);
  qurt_thread_attr_set_priority(&a, 128);
  qurt_thread_attr_set_name(&a, name);
  return qurt_thread_create(tid, &a, fn, 0);
}
static void stop(void) {
  int status;
  if (s.producer) {
    qurt_anysignal_set(&s.request, 2);
    qurt_thread_join(s.producer, &status);
    s.producer = 0;
  }
  for (unsigned i = 0; i < HELPERS; ++i) {
    if (helpers[i].tid) {
      qurt_anysignal_set(&helpers[i].request, 2);
      qurt_thread_join(helpers[i].tid, &status);
      helpers[i].tid = 0;
    }
    qurt_anysignal_destroy(&helpers[i].request);
    qurt_anysignal_destroy(&helpers[i].complete);
  }
  if (s.matrix) {
    qurt_anysignal_set(&s.matrix_request, 2);
    qurt_thread_join(s.matrix, &status);
    s.matrix = 0;
  }
  qurt_anysignal_destroy(&s.request);
  qurt_anysignal_destroy(&s.complete);
  qurt_anysignal_destroy(&s.matrix_request);
  qurt_anysignal_destroy(&s.matrix_complete);
  s.width = 0;
  s.ctx = 0;
}
int modular_hexagon_projection_invoke(unsigned long long handle,
                                      unsigned scalars, rpc_arg *args) {
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
  if (scalars == ((6u << 24) | (3u << 16) | (1u << 8))) {
    if (s.width || args[0].buf.len != 8)
      return 14;
    const unsigned *dims = args[0].buf.data;
    unsigned width = dims[0], k = dims[1];
    if (!width || width > 248320 || width % 32 || k < 128 || k > 6144 ||
        k % 128 || args[1].buf.len != (unsigned long long)width * k / 4 ||
        args[2].buf.len != (unsigned long long)width * k / 128 * 4 ||
        args[3].buf.len != 4)
      return 14;
    s.width = width;
    s.k = k;
    s.w = args[1].buf.data;
    s.scales = args[2].buf.data;
    qurt_anysignal_init(&s.request);
    qurt_anysignal_init(&s.complete);
    qurt_anysignal_init(&s.matrix_request);
    qurt_anysignal_init(&s.matrix_complete);
    int rc = start_thread(&s.matrix, matrix_worker, s.matrix_stack,
                          sizeof(s.matrix_stack), "mojo-hmx");
    for (unsigned i = 0; i < HELPERS; ++i) {
      qurt_anysignal_init(&helpers[i].request);
      qurt_anysignal_init(&helpers[i].complete);
    }
    for (unsigned i = 0; !rc && i < HELPERS; ++i) {
      qurt_thread_attr_t a;
      qurt_thread_attr_init(&a);
      qurt_thread_attr_set_stack_addr(&a, helpers[i].stack);
      qurt_thread_attr_set_stack_size(&a, sizeof(helpers[i].stack));
      qurt_thread_attr_set_priority(&a, 128);
      qurt_thread_attr_set_name(&a, "mojo-hvx-helper");
      rc = qurt_thread_create(&helpers[i].tid, &a, helper_worker, &helpers[i]);
      if (!rc) {
        qurt_anysignal_wait(&helpers[i].complete, 1);
        qurt_anysignal_clear(&helpers[i].complete, 1);
        rc = helpers[i].status;
      }
    }
    if (!rc)
      rc = start_thread(&s.producer, producer_worker, s.stack, sizeof(s.stack),
                        "mojo-hvx");
    if (!rc) {
      qurt_anysignal_wait(&s.complete, 1);
      qurt_anysignal_clear(&s.complete, 1);
      rc = s.status;
    }
    *(int *)args[3].buf.data = rc;
    if (rc)
      stop();
    return 0;
  }
  if (scalars == ((7u << 24) | (1u << 16) | (1u << 8))) {
    if (!s.width || !s.producer || args[0].buf.len != s.k * 2 ||
        args[1].buf.len != s.width * 4)
      return 14;
    s.x = args[0].buf.data;
    s.y = args[1].buf.data;
    qurt_anysignal_set(&s.request, 1);
    qurt_anysignal_wait(&s.complete, 1);
    qurt_anysignal_clear(&s.complete, 1);
    return s.status;
  }
  if (scalars == ((8u << 24) | (1u << 8)) &&
      args[0].buf.len == sizeof(s.debug)) {
    memcpy(args[0].buf.data, s.debug, sizeof(s.debug));
    return 0;
  }
  if (scalars == (9u << 24)) {
    if (s.width)
      stop();
    return 0;
  }
  return 14;
}
