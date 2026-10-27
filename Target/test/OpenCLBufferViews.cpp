// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
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
#define EXPECT(expr)                                                           \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr);             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)
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

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s OpenCLPlugin.so buffer_views.mcl\n",
                 argv[0]);
    return 2;
  }
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
  M_driver_context_t ctx = nullptr, other = nullptr;
  CHECK(api.context_create(device, &ctx));
  CHECK(api.context_create(device, &other));
  M_driver_queue_t queue = nullptr, reader = nullptr;
  CHECK(api.queue_create(ctx, &queue));
  CHECK(api.queue_create(ctx, &reader));
  M_driver_memory_t empty = nullptr;
  CHECK(api.memory_alloc_sync(ctx, 0, &empty));
  M_driver_memory_view emptyView{empty, 0, 0};
  CHECK(api.queue_copy_to_device(queue, &emptyView, nullptr));
  CHECK(api.queue_copy_from_device(queue, nullptr, &emptyView));
  CHECK(api.queue_fill(queue, &emptyView, 0, 1));
  CHECK(api.memory_free_sync(ctx, empty));
  FILE *file = std::fopen(argv[2], "rb");
  EXPECT(file);
  EXPECT(std::fseek(file, 0, SEEK_END) == 0);
  long size = std::ftell(file);
  EXPECT(size > 8 && size < (1 << 24));
  std::rewind(file);
  std::vector<char> bytes(size);
  EXPECT(std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size());
  std::fclose(file);
  EXPECT(std::memcmp(bytes.data(), "MOJOCL03", 8) == 0);
  M_driver_static_bundle bundle{
      {bytes.data(), bytes.size()}, "mojo-opencl-v3", 14};
  M_driver_runtime_bundle_t loaded = nullptr;
  // Catch file-type/magic mismatches and malformed manifests before JIT.
  bundle.mapped_data.size--;
  EXPECT(api.bundle_load(ctx, &bundle, nullptr, &loaded) ==
         M_DRIVER_STATUS_BAD_BUNDLE);
  bundle.mapped_data.size++;
  M::OpenCL::Bundle decoded;
  EXPECT(M::OpenCL::decodeBundle(bytes.data(), bytes.size(), decoded));
  decoded.requiredExtensions.push_back("cl_modular_test_missing_extension");
  auto unsupported = M::OpenCL::encodeBundle(decoded);
  bundle.mapped_data = {unsupported.data(), unsupported.size()};
  EXPECT(api.bundle_load(ctx, &bundle, nullptr, &loaded) !=
         M_DRIVER_STATUS_SUCCESS);
  char message[1024]{};
  CHECK(api.status_message(driver, M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1,
                           message, sizeof(message)));
  EXPECT(std::strstr(message, "cl_modular_test_missing_extension"));
  bundle.mapped_data = {bytes.data(), bytes.size()};
  M_driver_bundle_compilation_options opts{};
  opts.optimization_level = -1; // normal HAL default
  CHECK(api.bundle_load(ctx, &bundle, &opts, &loaded));
  M_driver_function_t fn = nullptr;
  CHECK(api.function_load(ctx, loaded, "view_copy", 9, &fn));
  M_driver_memory_t input = nullptr, output = nullptr, foreign = nullptr;
  CHECK(api.memory_alloc_sync(ctx, 64, &input));
  CHECK(api.memory_alloc_sync(ctx, 64, &output));
  CHECK(api.memory_alloc_sync(other, 64, &foreign));
  float values[16], actual[16];
  for (int i = 0; i < 16; ++i)
    values[i] = float(i);
  M_driver_memory_view wholeInput{input, 0, 64}, wholeOutput{output, 0, 64};
  CHECK(api.queue_copy_to_device(queue, &wholeInput, values));
  CHECK(api.queue_fill(queue, &wholeOutput, 0x7fc00000, 4));
  // Deliberately smaller than CL_DEVICE_MEM_BASE_ADDR_ALIGN: no cl subbuffer
  // alignment restriction applies to a scalar-aligned kernel byte offset.
  M_driver_memory_view src{input, 4, 20}, dst{output, 12, 20};
  M_driver_memory_view invalidView{
      reinterpret_cast<M_driver_memory_t>(uintptr_t(1)), 0, 4};
  EXPECT(api.queue_copy_from_device(queue, actual, &invalidView) ==
         M_DRIVER_STATUS_INVALID_ARG);
  int32_t count = 5;
  alignas(8) char sourcePacket[32]{}, destPacket[32]{}, scalarPacket[32]{};
  void *args[] = {sourcePacket, destPacket, scalarPacket};
  uint64_t sizes[] = {32, 32, 32};
  M_driver_queue_execute_config_t launch{};
  launch.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  launch.config.gpu.grid = {1, 1, 1};
  launch.config.gpu.block = {8, 1, 1};
  auto execute = [&] {
    std::memcpy(sourcePacket + 8, &src, sizeof(src));
    std::memcpy(destPacket + 8, &dst, sizeof(dst));
    std::memcpy(scalarPacket, &count, sizeof(count));
    return api.queue_execute(queue, fn, &launch, args, sizes, 3);
  };
  sizes[0] = 8;
  EXPECT(execute() == M_DRIVER_STATUS_INVALID_ARG);
  sizes[0] = 32;
  src.memory = foreign;
  EXPECT(execute() == M_DRIVER_STATUS_BAD_MEMORY);
  src.memory = reinterpret_cast<M_driver_memory_t>(uintptr_t(1));
  EXPECT(execute() == M_DRIVER_STATUS_BAD_MEMORY);
  src.memory = input;
  src.offset = UINT64_MAX;
  EXPECT(execute() == M_DRIVER_STATUS_BAD_MEMORY);
  src.offset = 4;
  src.size = 64;
  EXPECT(execute() == M_DRIVER_STATUS_BAD_MEMORY);
  src.size = 20;
  // The same native kernel can be reached through one combined logical
  // aggregate. Its pointer slots are descriptors in the sparse sidecar, not
  // addresses copied into the aggregate's scalar payload.
  decoded.requiredExtensions.clear();
  M::OpenCL::ArgumentABI aggregate{M::OpenCL::ArgumentABI::Aggregate, 96};
  aggregate.fields = {{0, M::OpenCL::ArgumentABI::BufferView, 24},
                      {8, M::OpenCL::ArgumentABI::BufferView, 24},
                      {16, M::OpenCL::ArgumentABI::Scalar, 4}};
  decoded.kernels[0].arguments = {aggregate};
  auto v3bytes = M::OpenCL::encodeBundle(decoded);
  M::OpenCL::Bundle roundtrip;
  EXPECT(M::OpenCL::decodeBundle(v3bytes.data(), v3bytes.size(), roundtrip));
  EXPECT(M::OpenCL::nativeArgumentCount(roundtrip.kernels[0]) == 5);
  for (uint32_t badOffset : {1u, 96u, UINT32_MAX}) {
    decoded.kernels[0].arguments[0].fields[1].offset = badOffset;
    auto invalid = M::OpenCL::encodeBundle(decoded);
    EXPECT(!M::OpenCL::decodeBundle(invalid.data(), invalid.size(), roundtrip));
  }
  M_driver_static_bundle v3bundle{
      {v3bytes.data(), v3bytes.size()}, "mojo-opencl-v3", 14};
  M_driver_runtime_bundle_t v3loaded = nullptr;
  CHECK(api.bundle_load(ctx, &v3bundle, &opts, &v3loaded));
  M_driver_function_t v3fn = nullptr;
  CHECK(api.function_load(ctx, v3loaded, "view_copy", 9, &v3fn));
  alignas(8) char packet[96]{};
  std::memcpy(packet + 16, &count, sizeof(count));
  std::memcpy(packet + 24, &src, sizeof(src));
  std::memcpy(packet + 48, &dst, sizeof(dst));
  void *v3args[] = {packet};
  uint64_t v3sizes[] = {32};
  EXPECT(api.queue_execute(queue, v3fn, &launch, v3args, v3sizes, 1) ==
         M_DRIVER_STATUS_INVALID_ARG);
  v3sizes[0] = sizeof(packet);
  auto badView = src;
  badView.memory = foreign;
  std::memcpy(packet + 24, &badView, sizeof(badView));
  EXPECT(api.queue_execute(queue, v3fn, &launch, v3args, v3sizes, 1) ==
         M_DRIVER_STATUS_BAD_MEMORY);
  std::memset(packet + 24, 0, sizeof(src));
  EXPECT(api.queue_execute(queue, v3fn, &launch, v3args, v3sizes, 1) ==
         M_DRIVER_STATUS_BAD_MEMORY);
  std::memcpy(packet + 24, &src, sizeof(src));
  CHECK(api.queue_execute(queue, v3fn, &launch, v3args, v3sizes, 1));
  CHECK(execute());
  // Native resources must remain valid for queued work after handle release.
  CHECK(api.memory_free_sync(ctx, input));
  EXPECT(execute() == M_DRIVER_STATUS_BAD_MEMORY);
  M_driver_event_t done = nullptr;
  CHECK(api.event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &done));
  CHECK(api.queue_record_event(queue, done));
  CHECK(api.queue_wait_for_events(reader, &done, 1));
  CHECK(api.queue_copy_from_device(reader, actual, &wholeOutput));
  for (int i = 0; i < 16; ++i) {
    if (i >= 3 && i < 8)
      EXPECT(actual[i] == values[i - 2] * 2 + 1);
    else {
      uint32_t bits;
      std::memcpy(&bits, &actual[i], 4);
      EXPECT(bits == 0x7fc00000);
    }
  }
  CHECK(api.memory_free_sync(ctx, output));
  CHECK(api.memory_free_sync(other, foreign));
  CHECK(api.event_destroy(ctx, done));
  CHECK(api.function_unload(ctx, v3fn));
  CHECK(api.bundle_unload(ctx, v3loaded));
  CHECK(api.function_unload(ctx, fn));
  CHECK(api.bundle_unload(ctx, loaded));
  CHECK(api.queue_destroy(ctx, reader));
  CHECK(api.queue_destroy(ctx, queue));
  CHECK(api.context_destroy(other));
  CHECK(api.context_destroy(ctx));
  CHECK(api.destroy(driver));
  dlclose(library);
  std::puts("PASS: compiled buffer views, unaligned subranges, cross-queue "
            "lifetime, v3 aggregate descriptors and invalid arguments");
}
