// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Single-session FastRPC transport for the Mojo packed HVX projector.
// The SDK's qurt_hvx.h requires size_t to be defined before including it.
#include <stddef.h>

#include "HAP_power.h"
#include "qurt_anysignal.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"
#include <stdint.h>
#include <string.h>

#ifndef MODULAR_HEXAGON_HVX_WORKERS
#define MODULAR_HEXAGON_HVX_WORKERS 8
#endif
#define HELPERS (MODULAR_HEXAGON_HVX_WORKERS - 1)
typedef union {
  struct {
    void *data;
    __SIZE_TYPE__ len;
  } buf;
  unsigned handle;
  unsigned long long handle64;
} rpc_arg;
extern void hexagon_prepare_lut(void *, void *, int);
extern void hexagon_hvx_project(void *, void *, void *, void *, int, int, int,
                                int);
static unsigned lut[6144 * 16] __attribute__((aligned(128)));
static struct worker {
  unsigned tid, first, count;
  int status;
  qurt_anysignal_t request, complete;
  unsigned char stack[16384] __attribute__((aligned(128)));
} helpers[MODULAR_HEXAGON_HVX_WORKERS];
static struct {
  unsigned width, k, producer;
  const void *weights, *scales, *input;
  void *output;
  int status;
  unsigned debug[16];
  qurt_anysignal_t request, complete;
  unsigned char stack[32768] __attribute__((aligned(128)));
} s;
static unsigned long long ticks(void) {
  unsigned long long t;
  __asm__ volatile("%0 = c31:30" : "=r"(t));
  return t;
}
static int power(void) {
  HAP_power_request_t p = {0};
  p.type = HAP_power_set_HVX;
  p.hvx.power_up = 1;
  int rc = HAP_power_set(&s, &p);
  if (rc)
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
static void project(unsigned first, unsigned count) {
  if (count)
    hexagon_hvx_project((void *)s.weights, (void *)s.scales, lut, s.output,
                        s.width, s.k, first, count);
}
static void helper_worker(void *data) {
  struct worker *h = data;
  h->status = qurt_hvx_lock(QURT_HVX_MODE_128B);
  qurt_anysignal_set(&h->complete, 1);
  if (!h->status) {
    for (;;) {
      unsigned command = qurt_anysignal_wait(&h->request, 3);
      qurt_anysignal_clear(&h->request, command);
      if (command & 2)
        break;
      project(h->first, h->count);
      qurt_anysignal_set(&h->complete, 1);
    }
    qurt_hvx_unlock();
  }
  qurt_thread_exit(0);
}
static void producer_worker(void *unused) {
  (void)unused;
  s.status = power();
  qurt_anysignal_set(&s.complete, 1);
  if (!s.status) {
    for (;;) {
      unsigned command = qurt_anysignal_wait(&s.request, 3);
      qurt_anysignal_clear(&s.request, command);
      if (command & 2)
        break;
      memset(s.debug, 0, sizeof(s.debug));
      unsigned long long start = ticks();
      s.status = qurt_hvx_lock(QURT_HVX_MODE_128B);
      if (!s.status) {
        hexagon_prepare_lut((void *)s.input, lut, s.k);
        s.debug[4] = (unsigned)(ticks() - start);
        unsigned chunk = ((s.width / 128 + MODULAR_HEXAGON_HVX_WORKERS - 1) /
                          MODULAR_HEXAGON_HVX_WORKERS) *
                         128;
        for (unsigned i = 0; i < HELPERS; ++i) {
          unsigned first = (i + 1) * chunk;
          helpers[i].first = first;
          helpers[i].count =
              first >= s.width
                  ? 0
                  : (s.width - first < chunk ? s.width - first : chunk);
          qurt_anysignal_set(&helpers[i].request, 1);
        }
        project(0, s.width < chunk ? s.width : chunk);
        for (unsigned i = 0; i < HELPERS; ++i) {
          qurt_anysignal_wait(&helpers[i].complete, 1);
          qurt_anysignal_clear(&helpers[i].complete, 1);
        }
        qurt_hvx_unlock();
      }
      s.debug[0] = (unsigned)(ticks() - start);
      qurt_anysignal_set(&s.complete, 1);
    }
  }
  // Removes both the HVX and DCVS votes belonging to this session.
  HAP_power_destroy(&s);
  qurt_thread_exit(0);
}
static int start_thread(unsigned *tid, void (*fn)(void *), void *arg,
                        void *stack, unsigned size) {
  qurt_thread_attr_t a;
  qurt_thread_attr_init(&a);
  qurt_thread_attr_set_stack_addr(&a, stack);
  qurt_thread_attr_set_stack_size(&a, size);
  qurt_thread_attr_set_priority(&a, 128);
  qurt_thread_attr_set_name(&a, "mojo-hvx-gemv");
  return qurt_thread_create(tid, &a, fn, arg);
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
  qurt_anysignal_destroy(&s.request);
  qurt_anysignal_destroy(&s.complete);
  s.width = 0;
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
    if (!width || width > 248320 || width % 128 || k < 128 || k > 6144 ||
        k % 128 || args[1].buf.len != (unsigned long long)width * k / 4 ||
        args[2].buf.len != (unsigned long long)width * k / 128 * 4 ||
        args[3].buf.len != 4)
      return 14;
    s.width = width;
    s.k = k;
    s.weights = args[1].buf.data;
    s.scales = args[2].buf.data;
    qurt_anysignal_init(&s.request);
    qurt_anysignal_init(&s.complete);
    for (unsigned i = 0; i < HELPERS; ++i) {
      qurt_anysignal_init(&helpers[i].request);
      qurt_anysignal_init(&helpers[i].complete);
    }
    int rc = 0;
    for (unsigned i = 0; !rc && i < HELPERS; ++i) {
      rc = start_thread(&helpers[i].tid, helper_worker, &helpers[i],
                        helpers[i].stack, sizeof(helpers[i].stack));
      if (!rc) {
        qurt_anysignal_wait(&helpers[i].complete, 1);
        qurt_anysignal_clear(&helpers[i].complete, 1);
        rc = helpers[i].status;
      }
    }
    if (!rc)
      rc = start_thread(&s.producer, producer_worker, 0, s.stack,
                        sizeof(s.stack));
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
    s.input = args[0].buf.data;
    s.output = args[1].buf.data;
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
