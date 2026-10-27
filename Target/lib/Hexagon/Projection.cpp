// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "Target/HexagonProjection.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
union Arg {
  struct {
    void *data;
    size_t len;
  } buf;
  uint32_t handle;
  uint64_t handle64;
};
struct Projection {
  bool bound = false;
  void *lib = nullptr;
  uint64_t handle = 0;
  uint32_t width = 0, k = 0;
  void *w = nullptr, *scales = nullptr, *input = nullptr, *output = nullptr;
  uint64_t rw = 0, rs = 0;
  size_t wb = 0, sb = 0;
  int (*session)(uint32_t, void *, uint32_t) = nullptr;
  int (*open)(const char *, uint64_t *) = nullptr;
  int (*invoke)(uint64_t, uint32_t, Arg *) = nullptr;
  int (*close)(uint64_t) = nullptr;
  int (*control)(uint64_t, uint32_t, void *, uint32_t) = nullptr;
  bool qosEnabled = false;
  void *(*alloc)(int, uint32_t, int) = nullptr;
  void (*free)(void *) = nullptr;
  int (*fd)(void *) = nullptr;
  int (*map)(int, int, int, uint64_t, size_t, uint64_t *) = nullptr;
  int (*unmap)(int, uint64_t, size_t) = nullptr;
  static void check(int status, const char *operation) {
    if (status)
      throw std::runtime_error(std::string(operation) + ": " +
                               std::to_string(status));
  }
  template <class T>
  void symbol(T &out, const char *name) {
    out = reinterpret_cast<T>(dlsym(lib, name));
    if (!out)
      throw std::runtime_error(std::string("FastRPC missing ") + name);
  }
  void *allocate(size_t bytes) {
    void *p = alloc(25, 1, int(bytes));
    if (!p)
      throw std::runtime_error("rpcmem allocation failed");
    return p;
  }
  Projection(uint32_t n, uint32_t reduction, const uint32_t *packed,
             const float *params)
      : width(n), k(reduction) {
    try {
      if (!width || width > 248320 || width % 32 || k < 128 || k > 6144 ||
          k % 128)
        throw std::runtime_error("Unsupported Hexagon projection shape");
#ifdef MODULAR_HEXAGON_HVX
      if (width % 128)
        throw std::runtime_error(
            "HVX projection width must be a multiple of 128");
#endif
      lib = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
      if (!lib)
        throw std::runtime_error(dlerror());
      symbol(session, "remote_session_control");
      symbol(open, "remote_handle64_open");
      symbol(invoke, "remote_handle64_invoke");
      symbol(close, "remote_handle64_close");
      symbol(alloc, "rpcmem_alloc");
      symbol(free, "rpcmem_free");
      symbol(fd, "rpcmem_to_fd");
      symbol(map, "remote_mem_map");
      symbol(unmap, "remote_mem_unmap");
      struct {
        int domain, enabled;
      } unsigned_pd{3, 1};
      check(session(2, &unsigned_pd, sizeof(unsigned_pd)),
            "unsigned CDSP session");
      check(open("file:///"
                 "libmodular_hexagon.so?modular_hexagon_projection_invoke&_"
                 "modver=1.0&_dom=cdsp",
                 &handle),
            "open Mojo projector");
      if (const char *q = std::getenv("MODULAR_HEXAGON_QOS")) {
        // SDK DSPRPC_CONTROL_LATENCY / RPC_PM_QOS: process-local CPU wakeup
        // latency vote. Polling QoS is unsupported on the tested phone.
        if (std::strcmp(q, "0") && std::strcmp(q, "1"))
          throw std::runtime_error("MODULAR_HEXAGON_QOS must be 0 or 1");
        symbol(control, "remote_handle64_control");
        struct {
          uint32_t enable, latency;
        } qos{uint32_t(*q - '0'), 100};
        check(control(handle, 1, &qos, sizeof(qos)), "FastRPC latency control");
        qosEnabled = qos.enable != 0;
      }
      wb = size_t(width) * k / 4;
      sb = size_t(width) * (k / 128) * 4;
      w = allocate(wb);
      scales = allocate(sb);
      input = allocate(k * 2);
      output = allocate(width * 4);
      // Block output channels once so each HVX worker streams packed words.
#ifdef MODULAR_HEXAGON_HVX
      auto *dst = static_cast<uint8_t *>(w);
      for (uint32_t block = 0; block < width / 128; ++block)
        for (uint32_t byte = 0; byte < k / 4; ++byte)
          for (uint32_t col = 0; col < 128; ++col)
            dst[(size_t(block) * (k / 4) + byte) * 128 + col] =
                uint8_t(packed[size_t(byte / 4) * width + block * 128 + col] >>
                        ((byte % 4) * 8));
#else
      auto *dst = static_cast<uint32_t *>(w);
      for (uint32_t block = 0; block < width / 32; ++block)
        for (uint32_t word = 0; word < k / 16; ++word)
          std::memcpy(dst + (size_t(block) * (k / 16) + word) * 32,
                      packed + size_t(word) * width + block * 32, 128);
#endif
      for (size_t i = 0; i < sb / 4; ++i) {
        float scale, bias;
        std::memcpy(&scale, reinterpret_cast<const char *>(params) + i * 8, 4);
        std::memcpy(&bias, reinterpret_cast<const char *>(params) + i * 8 + 4,
                    4);
        if (!std::isfinite(scale) || bias != -scale)
          throw std::runtime_error(
              "Expected finite centered Bonsai affine groups");
        static_cast<float *>(scales)[i] = scale;
      }
      // Map once; passing these buffers in bind performs initial cache
      // maintenance.
      check(map(3, fd(w), 0, reinterpret_cast<uint64_t>(w), wb, &rw),
            "map packed weights");
      check(map(3, fd(scales), 0, reinterpret_cast<uint64_t>(scales), sb, &rs),
            "map group scales");
      uint32_t shape[] = {width, k};
      int status = -1;
      Arg args[4]{};
      args[0].buf = {shape, sizeof(shape)};
      args[1].buf = {w, wb};
      args[2].buf = {scales, sb};
      args[3].buf = {&status, sizeof(status)};
      check(invoke(handle, (6u << 24) | (3u << 16) | (1u << 8), args),
            "bind resident weights");
      check(status, "initialize HVX/HMX workers");
      bound = true;
    } catch (...) {
      cleanup();
      throw;
    }
  }
  void cleanup() {
    if (bound && handle && invoke) {
      Arg unused{};
      invoke(handle, 9u << 24, &unused);
      bound = false;
    }
    if (rw && unmap)
      unmap(3, rw, wb);
    if (rs && unmap)
      unmap(3, rs, sb);
    if (qosEnabled && handle && control) {
      struct {
        uint32_t enable, latency;
      } qos{0, 100};
      control(handle, 1, &qos, sizeof(qos));
      qosEnabled = false;
    }
    if (handle && close)
      close(handle);
    handle = 0;
    if (free) {
      for (void *p : {w, scales, input, output})
        if (p)
          free(p);
    }
    w = scales = input = output = nullptr;
    if (lib)
      dlclose(lib);
    lib = nullptr;
  }
  ~Projection() { cleanup(); }
  void execute(const uint16_t *x, float *y) {
    std::memcpy(input, x, k * 2);
    Arg args[2]{};
    args[0].buf = {input, k * 2};
    args[1].buf = {output, width * 4};
    check(invoke(handle, (7u << 24) | (1u << 16) | (1u << 8), args),
          "Mojo HMX projection");
    if (std::getenv("MODULAR_HEXAGON_PROFILE")) {
      unsigned debug[16]{};
      Arg d{};
      d.buf = {debug, sizeof(debug)};
      check(invoke(handle, (8u << 24) | (1u << 8), &d), "profile");
#ifdef MODULAR_HEXAGON_HVX
      std::fprintf(stderr, "HEXAGON_PROFILE hvx_ms=%.3f lookup_ms=%.3f\n",
                   debug[0] / 19200., debug[4] / 19200.);
#else
      std::fprintf(stderr,
                   "HEXAGON_PROFILE unpack_ms=%.3f hmx_ms=%.3f reduce_ms=%.3f "
                   "wait_ms=%.3f activation_ms=%.3f\n",
                   debug[0] / 19200., debug[1] / 19200., debug[2] / 19200.,
                   debug[3] / 19200., debug[4] / 19200.);
#endif
    }
    std::memcpy(y, output, width * 4);
  }
};
void error(char *dst, size_t size, const char *message) {
  if (dst && size)
    std::snprintf(dst, size, "%s", message);
}
} // namespace
extern "C" int M_hexagon_projection_create(uint32_t width, uint32_t k,
                                           const uint32_t *packed,
                                           const float *affine,
                                           M_hexagon_projection_t *out,
                                           char *message, size_t size) {
  if (!out || !packed || !affine) {
    error(message, size, "Null projector argument");
    return 1;
  }
  *out = nullptr;
  try {
    *out = new Projection(width, k, packed, affine);
    return 0;
  } catch (const std::exception &e) {
    error(message, size, e.what());
    return 1;
  }
}
extern "C" int M_hexagon_projection_execute(M_hexagon_projection_t handle,
                                            const uint16_t *x, float *y,
                                            char *message, size_t size) {
  if (!handle || !x || !y) {
    error(message, size, "Null execution argument");
    return 1;
  }
  try {
    static_cast<Projection *>(handle)->execute(x, y);
    return 0;
  } catch (const std::exception &e) {
    error(message, size, e.what());
    return 1;
  }
}
extern "C" void M_hexagon_projection_destroy(M_hexagon_projection_t handle) {
  delete static_cast<Projection *>(handle);
}
