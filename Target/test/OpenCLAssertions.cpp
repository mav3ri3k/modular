// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Exercise a compiler-produced assertion bundle through the standard SDK HAL.
#include "HAL/M_driver_api.h"
#include "Target/OpenCLBundle.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

struct Api {
#define M_DRIVER_HAL_SYMBOL(name) decltype(&M_driver_##name) name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
};
static Api api;
static M_driver_driver_t driver = nullptr;
static constexpr auto assertion = M_DRIVER_STATUS_CUSTOM_CODES_BASE - 2;
#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)
static void check(M_driver_status_t status, const char *expression) {
  if (status == M_DRIVER_STATUS_SUCCESS)
    return;
  char message[1024]{};
  api.status_message(driver, status, message, sizeof(message));
  std::fprintf(stderr, "FAIL %s: %lld %s\n", expression, (long long)status,
               message);
  std::exit(1);
}
#define CHECK(expression) check(expression, #expression)

int main(int argc, char **argv) {
  EXPECT(argc == 3);
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  EXPECT(library);
#define M_DRIVER_HAL_SYMBOL(name)                                              \
  api.name =                                                                   \
      reinterpret_cast<decltype(api.name)>(dlsym(library, "M_driver_" #name)); \
  EXPECT(api.name);
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
  M_driver_version version{M_DRIVER_INTERFACE_VERSION_MAJOR,
                           M_DRIVER_INTERFACE_VERSION_MINOR,
                           M_DRIVER_INTERFACE_VERSION_PATCH};
  CHECK(api.create(&version, &driver));
  M_driver_device_t device = nullptr;
  CHECK(api.device_get(driver, 0, &device));
  M_driver_context_t ctx = nullptr;
  CHECK(api.context_create(device, &ctx));
  FILE *file = std::fopen(argv[2], "rb");
  EXPECT(file && std::fseek(file, 0, SEEK_END) == 0);
  long size = std::ftell(file);
  EXPECT(size > 8 && size < (1 << 24));
  std::rewind(file);
  std::vector<char> bytes(size);
  EXPECT(std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size());
  std::fclose(file);
  M::OpenCL::Bundle decoded;
  EXPECT(M::OpenCL::decodeBundle(bytes.data(), bytes.size(), decoded));
  EXPECT(decoded.kernels.size() == 2);
  for (const auto &kernel : decoded.kernels)
    if (kernel.name == "formatted_assertion") {
      EXPECT(kernel.deviceErrors);
      EXPECT(M::OpenCL::nativeArgumentCount(kernel) == 3);
    }
  M_driver_static_bundle bundle{
      {bytes.data(), bytes.size()}, "mojo-opencl-v3", 14};
  M_driver_runtime_bundle_t loaded = nullptr;
  CHECK(api.bundle_load(ctx, &bundle, nullptr, &loaded));
  M_driver_function_t function = nullptr;
  CHECK(api.function_load(ctx, loaded, "formatted_assertion", 19, &function));
  M_driver_memory_t memory = nullptr;
  CHECK(api.memory_alloc_sync(ctx, sizeof(int32_t), &memory));
  M_driver_memory_view view{memory, 0, sizeof(int32_t)};
  alignas(8) char packet[32]{};
  std::memcpy(packet + 8, &view, sizeof(view));
  void *arguments[] = {packet};
  uint64_t sizes[] = {sizeof(packet)};
  M_driver_queue_execute_config_t launch{};
  launch.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  launch.config.gpu.grid = {1, 1, 1};
  launch.config.gpu.block = {1, 1, 1};
  // A null optional is a real OpenCL null pointer, not an allocation handle.
  M_driver_function_t nullable = nullptr;
  CHECK(api.function_load(ctx, loaded, "nullable_input", 14, &nullable));
  alignas(8) char nullPacket[32]{};
  void *nullableArgs[] = {packet, nullPacket};
  uint64_t nullableSizes[] = {sizeof(packet), sizeof(nullPacket)};
  M_driver_queue_t nullableQueue = nullptr;
  CHECK(api.queue_create(ctx, &nullableQueue));
  CHECK(api.queue_execute(nullableQueue, nullable, &launch, nullableArgs,
                          nullableSizes, 2));
  int32_t nullableResult = -1;
  CHECK(api.queue_copy_from_device(nullableQueue, &nullableResult, &view));
  EXPECT(nullableResult == 0);
  std::memcpy(nullPacket + 8, &view, sizeof(view));
  CHECK(api.queue_execute(nullableQueue, nullable, &launch, nullableArgs,
                          nullableSizes, 2));
  CHECK(api.queue_copy_from_device(nullableQueue, &nullableResult, &view));
  EXPECT(nullableResult == 1);
  M_driver_memory_view malformed{nullptr, 1, 0};
  std::memcpy(nullPacket + 8, &malformed, sizeof(malformed));
  EXPECT(api.queue_execute(nullableQueue, nullable, &launch, nullableArgs,
                           nullableSizes, 2) == M_DRIVER_STATUS_BAD_MEMORY);
  malformed = {nullptr, 0, 1};
  std::memcpy(nullPacket + 8, &malformed, sizeof(malformed));
  EXPECT(api.queue_execute(nullableQueue, nullable, &launch, nullableArgs,
                           nullableSizes, 2) == M_DRIVER_STATUS_BAD_MEMORY);
  CHECK(api.queue_destroy(ctx, nullableQueue));
  CHECK(api.function_unload(ctx, nullable));
  std::puts("PASS: compiler preserves optional null checks; malformed null "
            "views rejected");
  auto enqueue = [&](M_driver_queue_t queue, int32_t value) {
    CHECK(api.queue_copy_to_device(queue, &view, &value));
    CHECK(api.queue_execute(queue, function, &launch, arguments, sizes, 1));
  };
  M_driver_queue_t queue = nullptr;
  CHECK(api.queue_create(ctx, &queue));
  enqueue(queue, 7);
  M_driver_event_t event = nullptr;
  CHECK(api.event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
  CHECK(api.queue_record_event(queue, event));
  CHECK(api.event_synchronize(ctx, event));
  int32_t value = 0;
  CHECK(api.queue_copy_from_device(queue, &value, &view));
  EXPECT(value == 8);
  CHECK(api.queue_synchronize(queue));
  CHECK(api.event_destroy(ctx, event));
  CHECK(api.queue_destroy(ctx, queue));
  std::puts("PASS: assertion success path is asynchronous and produces "
            "expected output");

  CHECK(api.queue_create(ctx, &queue));
  enqueue(queue, 0);
  CHECK(api.event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
  CHECK(api.queue_record_event(queue, event));
  EXPECT(api.event_synchronize(ctx, event) == assertion);
  bool ready = false;
  EXPECT(api.is_event_ready(ctx, event, &ready) == assertion && ready);
  EXPECT(api.queue_synchronize(queue) == assertion);
  EXPECT(api.queue_copy_from_device(queue, &value, &view) == assertion);
  EXPECT(api.queue_destroy(ctx, queue) == assertion);
  // Event/readback lifetime is independent of the source queue.
  EXPECT(api.event_synchronize(ctx, event) == assertion);
  CHECK(api.event_destroy(ctx, event));
  CHECK(api.copy_from_device_sync(ctx, &value, &view));
  EXPECT(value == 0); // The failed assertion prevented the increment.
  char message[1024]{};
  CHECK(api.status_message(driver, assertion, message, sizeof(message)));
  EXPECT(std::strstr(message, "device assertion failed"));
  std::puts(
      "PASS: failure reaches event query/wait, queue sync and blocking read");

  M_driver_queue_t dependent = nullptr, healthy = nullptr;
  CHECK(api.queue_create(ctx, &queue));
  CHECK(api.queue_create(ctx, &dependent));
  CHECK(api.queue_create(ctx, &healthy));
  enqueue(queue, 0);
  CHECK(api.event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
  CHECK(api.queue_record_event(queue, event));
  CHECK(api.queue_wait_for_events(dependent, &event, 1));
  // Re-recording must not replace the dependent queue's captured failure.
  CHECK(api.queue_record_event(healthy, event));
  CHECK(api.event_synchronize(ctx, event));
  M_driver_event_t downstream = nullptr;
  CHECK(api.event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &downstream));
  CHECK(api.queue_record_event(dependent, downstream));
  CHECK(api.event_destroy(ctx, event));
  EXPECT(api.event_synchronize(ctx, downstream) == assertion);
  EXPECT(api.queue_copy_from_device(dependent, &value, &view) == assertion);
  CHECK(api.event_destroy(ctx, downstream));
  EXPECT(api.queue_destroy(ctx, queue) == assertion);
  EXPECT(api.queue_destroy(ctx, dependent) == assertion);
  CHECK(api.queue_destroy(ctx, healthy));
  std::puts(
      "PASS: cross-queue failures survive event re-recording/destruction");

  CHECK(api.queue_create(ctx, &queue));
  enqueue(queue, 0);
  CHECK(api.event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
  CHECK(api.queue_record_event(queue, event));
  CHECK(api.event_destroy(ctx, event));
  EXPECT(api.queue_synchronize(queue) == assertion);
  EXPECT(api.queue_destroy(ctx, queue) == assertion);
  CHECK(api.memory_free_sync(ctx, memory));
  CHECK(api.function_unload(ctx, function));
  CHECK(api.bundle_unload(ctx, loaded));
  CHECK(api.context_destroy(ctx));
  CHECK(api.destroy(driver));
  dlclose(library);
  std::puts(
      "PASS: early event destruction retains asynchronous readback storage");
}
