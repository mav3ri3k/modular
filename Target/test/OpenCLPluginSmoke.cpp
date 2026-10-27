// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "HAL/M_driver_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <vector>

struct Api {
#define M_DRIVER_HAL_SYMBOL(name) decltype(&M_driver_##name) name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
};
static Api api;
static M_driver_driver_t driver = nullptr;
static void check(M_driver_status_t status, const char *expression) {
  if (status == M_DRIVER_STATUS_SUCCESS)
    return;
  char message[4096] = {};
  api.status_message(driver, status, message, sizeof(message));
  std::fprintf(stderr, "FAIL %s: %lld %s\n", expression, (long long)status,
               message);
  std::exit(1);
}
#define CHECK(expr) check((expr), #expr)
#define EXPECT(expr)                                                           \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::fprintf(stderr, "FAIL %s\n", #expr);                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
static constexpr char source[] = R"cl(
__kernel void matmul(__global const float *a, __global const float *b,
                     __global float *c, int m, int n, int k) {
  int j = get_global_id(0), i = get_global_id(1);
  if (i >= m || j >= n) return;
  float sum = 0;
  for (int p = 0; p < k; ++p) sum += a[i*k+p] * b[p*n+j];
  c[i*n+j] = sum;
}
)cl";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s /path/OpenCLPlugin.so\n", argv[0]);
    return 2;
  }
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    std::fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
#define M_DRIVER_HAL_SYMBOL(name)                                              \
  api.name =                                                                   \
      reinterpret_cast<decltype(api.name)>(dlsym(library, "M_driver_" #name)); \
  EXPECT(api.name);
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
  std::puts("PASS: all SDK HAL symbols resolved");
  M_driver_version version{99, 0, 0};
  EXPECT(api.create(&version, &driver) == M_DRIVER_STATUS_INCOMPATIBLE_VERSION);
  version = {0, 1, 0};
  CHECK(api.create(&version, &driver));
  const char *backend = nullptr;
  CHECK(api.property(driver, M_DRIVER_PROPERTY_API, &backend));
  EXPECT(std::strcmp(backend, "OpenCL") == 0);
  int64_t count = 0;
  CHECK(api.device_count(driver, &count));
  EXPECT(count > 0);
  M_driver_device_t device = nullptr;
  EXPECT(api.device_get(driver, count, &device) == M_DRIVER_STATUS_BAD_DEVICE);
  CHECK(api.device_get(driver, 0, &device));
  char name[256] = {};
  CHECK(api.device_property(device, M_DRIVER_DEVICE_PROPERTY_NAME, name));
  std::printf("OpenCL GPU: %s; devices=%lld\n", name, (long long)count);
  M_driver_dlpack_device dlpack{};
  CHECK(api.device_property(device, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE,
                            &dlpack));
  EXPECT(dlpack.device_type == 4);
  M_driver_context_t context = nullptr, otherContext = nullptr;
  CHECK(api.context_create(device, &context));
  CHECK(api.context_create(device, &otherContext));
  M_driver_queue_t queue = nullptr, reader = nullptr;
  CHECK(api.queue_create(context, &queue));
  CHECK(api.queue_create(context, &reader));
  bool stream = false;
  CHECK(api.queue_is_stream(queue, &stream));
  EXPECT(stream);
  for (uint64_t size : {uint64_t(0), uint64_t(64)}) {
    M_driver_memory_t pinned = nullptr;
    CHECK(api.memory_alloc_pinned(context, size, &pinned));
    uint64_t address = 0;
    CHECK(api.memory_property(pinned, M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS,
                              &address));
    EXPECT(address != 0);
    if (size)
      std::memset(reinterpret_cast<void *>(uintptr_t(address)), 0x5a, size);
    EXPECT(api.memory_free_sync(context, pinned) == M_DRIVER_STATUS_BAD_MEMORY);
    EXPECT(api.memory_free_pinned(otherContext, pinned) ==
           M_DRIVER_STATUS_BAD_MEMORY);
    M_driver_memory_view hostView{pinned, 0, size};
    CHECK(api.queue_fill(queue, &hostView, 0x37, 1));
    std::vector<uint8_t> copied(size);
    CHECK(api.queue_copy_from_device(queue, copied.data(), &hostView));
    for (auto byte : copied)
      EXPECT(byte == 0x37);
    CHECK(api.memory_free_pinned(context, pinned));
  }
  std::puts(
      "PASS: pinned host mapping, zero-size allocation and handle validation");
  M_driver_event_t event = nullptr;
  CHECK(api.event_create(context, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
  EXPECT(api.event_synchronize(context, event) == M_DRIVER_STATUS_BAD_EVENT);
  M_driver_static_bundle bundle{{source, sizeof(source) - 1}, "wrong", 5};
  M_driver_runtime_bundle_t loaded = nullptr;
  EXPECT(api.bundle_load(context, &bundle, nullptr, &loaded) ==
         M_DRIVER_STATUS_INCOMPATIBLE);
  bundle.file_type = "opencl-source";
  bundle.file_type_len = 13;
  const char *invalid = "__kernel void broken( {";
  bundle.mapped_data = {invalid, std::strlen(invalid)};
  EXPECT(api.bundle_load(context, &bundle, nullptr, &loaded) !=
         M_DRIVER_STATUS_SUCCESS);
  char buildError[4096] = {};
  CHECK(api.status_message(driver, M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1,
                           buildError, sizeof(buildError)));
  EXPECT(std::strstr(buildError, "error"));
  bundle.mapped_data = {source, sizeof(source) - 1};
  CHECK(api.bundle_load(context, &bundle, nullptr, &loaded));
  M_driver_function_t function = nullptr;
  CHECK(api.function_load(context, loaded, "matmul", 6, &function));
  for (auto shape : {std::vector<int>{2, 2, 3},
                     {1, 7, 5},
                     {5, 1, 9},
                     {8, 6, 7},
                     {17, 19, 23},
                     {2, 3, 0}}) {
    int m = shape[0], n = shape[1], k = shape[2];
    std::vector<float> a(std::max(1, m * k)), b(std::max(1, k * n)), c(m * n),
        expected(m * n);
    for (int i = 0; i < m * k; ++i)
      a[i] = float((i * 7) % 13 - 6) / 4;
    for (int i = 0; i < k * n; ++i)
      b[i] = float((i * 3) % 11 - 5) / 8;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        for (int p = 0; p < k; ++p)
          expected[i * n + j] += a[i * k + p] * b[p * n + j];
    M_driver_memory_t memories[3] = {};
    M_driver_memory_view views[3];
    uint64_t addresses[3] = {};
    const uint64_t sizes[] = {a.size() * 4, b.size() * 4, c.size() * 4};
    for (int i = 0; i < 3; ++i) {
      CHECK(api.memory_alloc_sync(context, sizes[i] + 16, &memories[i]));
      views[i] = {memories[i], 0, sizes[i]};
      CHECK(api.memory_property(memories[i], M_DRIVER_MEMORY_PROPERTY_ADDRESS,
                                &addresses[i]));
    }
    EXPECT(api.memory_free_sync(otherContext, memories[0]) ==
           M_DRIVER_STATUS_BAD_MEMORY);
    M_driver_memory_view invalidView{memories[0], sizes[0] + 16, 1};
    EXPECT(api.queue_copy_to_device(queue, &invalidView, a.data()) ==
           M_DRIVER_STATUS_INVALID_ARG);
    CHECK(api.queue_copy_to_device(queue, &views[0], a.data()));
    CHECK(api.queue_copy_to_device(queue, &views[1], b.data()));
    // Offset copies are supported even though kernel arguments use opaque
    // cl_mem handles.
    float sentinel = 42, copied = 0;
    M_driver_memory_view tail{memories[2], sizes[2], sizeof(float)};
    CHECK(api.queue_copy_to_device(queue, &tail, &sentinel));
    M_driver_queue_execute_config_t config{};
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
    config.config.gpu.grid = {uint32_t((n + 3) / 4), uint32_t((m + 3) / 4), 1};
    config.config.gpu.block = {4, 4, 1};
    void *args[] = {&addresses[0], &addresses[1], &addresses[2], &m, &n, &k};

    uint64_t argSizes[] = {8, 8, 8, 4, 4, 4};
    EXPECT(api.queue_execute(queue, function, &config, args, argSizes, 5) ==
           M_DRIVER_STATUS_INVALID_ARG);
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_NPU;
    EXPECT(api.queue_execute(queue, function, &config, args, argSizes, 6) ==
           M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED);
    config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
    config.config.gpu.shared_mem_bytes = 1;
    EXPECT(api.queue_execute(queue, function, &config, args, argSizes, 6) ==
           M_DRIVER_STATUS_UNSUPPORTED);
    config.config.gpu.shared_mem_bytes = 0;

    CHECK(api.queue_execute(queue, function, &config, args, argSizes, 6));
    CHECK(api.queue_record_event(queue, event));
    CHECK(api.queue_wait_for_events(reader, &event, 1));
    CHECK(api.queue_copy_from_device(reader, c.data(), &views[2]));
    CHECK(api.queue_copy_from_device(reader, &copied, &tail));
    EXPECT(copied == sentinel);
    CHECK(api.event_synchronize(context, event));
    bool ready = false;
    CHECK(api.is_event_ready(context, event, &ready));
    EXPECT(ready);
    for (size_t i = 0; i < c.size(); ++i)
      EXPECT(std::abs(c[i] - expected[i]) < 1e-4f);
    CHECK(api.queue_fill(reader, &tail, 0x12345678, 4));
    uint32_t pattern = 0;
    CHECK(api.queue_copy_from_device(reader, &pattern, &tail));
    EXPECT(pattern == 0x12345678);
    EXPECT(api.queue_fill(reader, &tail, 0, 3) == M_DRIVER_STATUS_INVALID_ARG);
    M_driver_memory_view first{memories[2], 0, sizeof(float)};
    CHECK(api.queue_copy_intra_device(reader, &tail, &first));
    CHECK(api.queue_copy_from_device(reader, &copied, &tail));
    EXPECT(copied == c[0]);
    for (int i = 0; i < 3; ++i)
      CHECK(api.memory_free_sync(context, memories[i]));
    std::printf(
        "PASS: HAL GPU matmul %dx%d @ %dx%d, cross-queue event and bounds\n", m,
        k, k, n);
  }

  EXPECT(api.context_set_current(context) == M_DRIVER_STATUS_UNIMPLEMENTED);
  CHECK(api.queue_synchronize(queue));
  CHECK(api.queue_synchronize(reader));
  CHECK(api.function_unload(context, function));
  CHECK(api.bundle_unload(context, loaded));
  CHECK(api.event_destroy(context, event));
  CHECK(api.queue_destroy(context, reader));
  CHECK(api.queue_destroy(context, queue));
  CHECK(api.context_destroy(otherContext));
  CHECK(api.context_destroy(context));
  CHECK(api.destroy(driver));
  dlclose(library);
  std::puts("PASS: OpenCL HAL lifecycle and negative tests");
}
