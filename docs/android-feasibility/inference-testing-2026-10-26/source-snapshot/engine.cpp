// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Checkpoint I/O and embedding only; Mojo owns all device resources.
#include "engine.h"
#include "Target/HexagonProjection.h"
#include "model_spec.h"
#include "sampling.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

extern "C" {
void *android_model_create();
void android_model_destroy(void *);
int64_t android_model_allocate(void *, int32_t, int64_t, const void *, int32_t);
int64_t android_model_alias_weight(void *, int32_t);
int64_t android_model_synchronize(void *);
int64_t android_model_zero(void *, int32_t);
int64_t android_model_read(void *, int32_t, void *, int64_t);
int64_t android_model_read_async(void *, int32_t, void *, int64_t);
int64_t android_model_name(void *, char *, int64_t);
int64_t android_model_forward(void *, int32_t, int32_t, int32_t);
int64_t android_model_forward_with_head_input(void *, int32_t, int32_t, int32_t,
                                              void *);
void android_model_set_head_columns(void *, int32_t);
}
static void require(bool ok, const char *what) {
  if (!ok)
    throw std::runtime_error(what);
}
static void check(int64_t status) {
  require(status == 0, "Mojo model operation failed (see diagnostic)");
}
template <class T>
static T read(std::istream &file) {
  T value{};
  require(bool(file.read(reinterpret_cast<char *>(&value), sizeof(value))),
          "Truncated model");
  return value;
}
struct HexagonHead {
  void *library = nullptr;
  M_hexagon_projection_t projector = nullptr;
  decltype(&M_hexagon_projection_create) create = nullptr;
  decltype(&M_hexagon_projection_execute) execute = nullptr;
  decltype(&M_hexagon_projection_destroy) destroy = nullptr;
  std::vector<char> packed;
  std::vector<uint16_t> input;
  std::vector<float> logits;
  bool ready = false;
  uint32_t gpuColumns = 0;
  explicit HexagonHead(const char *path) {
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error(dlerror());
    create = reinterpret_cast<decltype(create)>(
        dlsym(library, "M_hexagon_projection_create"));
    execute = reinterpret_cast<decltype(execute)>(
        dlsym(library, "M_hexagon_projection_execute"));
    destroy = reinterpret_cast<decltype(destroy)>(
        dlsym(library, "M_hexagon_projection_destroy"));
    if (!create || !execute || !destroy) {
      dlclose(library);
      library = nullptr;
      throw std::runtime_error("Missing Hexagon projection API");
    }
  }
  ~HexagonHead() {
    if (projector)
      destroy(projector);
    if (library)
      dlclose(library);
  }
};

struct AndroidLlmEngine::Impl {
  struct StateDeleter {
    void operator()(void *p) const { android_model_destroy(p); }
  };
  std::unique_ptr<void, StateDeleter> state;
  std::unique_ptr<HexagonHead> hexagon;
  std::vector<float> greedyCandidates;
  bool greedyReadPending = false;
  uint32_t capacity = 0;
  int32_t position = 0;
  std::string deviceName;
  Impl(const std::string &plugin, const std::string &weightsPath) {
    std::string plugins = "OpenCL@" + plugin;
    require(setenv("MODULAR_DRIVER_PLUGINS", plugins.c_str(), 1) == 0,
            "Cannot configure driver plugin");
    state.reset(android_model_create());
    require(bool(state), "Cannot create Mojo DeviceContext");
    char name[256]{};
    check(android_model_name(state.get(), name, sizeof(name)));
    deviceName = name;
    std::ifstream model(weightsPath, std::ios::binary);
    char magic[8]{};
    require(bool(model.read(magic, 8)) &&
                std::memcmp(magic, modelMagic, 8) == 0,
            "Bad model magic");
    require(read<uint32_t>(model) == weightCount, "Bad tensor count");
    capacity = read<uint32_t>(model);
    require(capacity >= 1 && capacity <= 8192, "Bad cache capacity");
    uint64_t totalBytes = 0;
#ifdef BONSAI_MODEL
    if (const char *path = std::getenv("MODULAR_HEXAGON_HEAD")) {
      if (*path)
        hexagon = std::make_unique<HexagonHead>(path);
    }
#endif
    std::map<int, std::vector<char>> sharedSigns;
    for (int i = 0; i < weightCount; ++i) {
      uint64_t size = read<uint64_t>(model);
      require(size == weightBytes(i, capacity), "Bad tensor size");
      std::vector<char> tensor(size);
      require(bool(model.read(tensor.data(), size)), "Truncated tensor");
      int signGroup = sharedSignGroup(i);
      if (signGroup >= 0) {
        auto found = sharedSigns.find(signGroup);
        if (found == sharedSigns.end())
          sharedSigns.emplace(signGroup, tensor);
        else
          require(found->second == tensor, "Projection sign vectors differ");
      }
      if (centeredParams(i)) {
        for (uint64_t offset = 0; offset < size; offset += 8) {
          float scale, bias;
          std::memcpy(&scale, tensor.data() + offset, 4);
          std::memcpy(&bias, tensor.data() + offset + 4, 4);
          require(std::isfinite(scale) && bias == -scale,
                  "Packed group is not centered");
        }
      }
#ifdef BONSAI_MODEL
      if (hexagon && i == 0)
        hexagon->packed = tensor;
      if (hexagon && i == 1) {
        auto &head = *hexagon;
        char error[512]{};
        uint32_t columns = vocabulary;
        if (const char *env = std::getenv("MODULAR_HEXAGON_HEAD_COLUMNS")) {
          auto requested = std::stoul(env);
          require(requested >= 128 && requested <= vocabulary &&
                      requested % 128 == 0,
                  "NPU head columns must be a multiple of 128 within the "
                  "vocabulary");
          columns = uint32_t(requested);
        }
        head.gpuColumns = vocabulary - columns;
        const auto *packed =
            reinterpret_cast<const uint32_t *>(head.packed.data());
        const auto *affine = reinterpret_cast<const float *>(tensor.data());
        std::vector<uint32_t> subset;
        std::vector<float> parameters;
        if (head.gpuColumns) {
          subset.resize(size_t(columns) * 2048 / 16);
          parameters.resize(size_t(columns) * 16 * 2);
          for (unsigned word = 0; word < 128; ++word)
            std::memcpy(subset.data() + size_t(word) * columns,
                        packed + size_t(word) * vocabulary + head.gpuColumns,
                        size_t(columns) * 4);
          for (unsigned group = 0; group < 16; ++group)
            std::memcpy(parameters.data() + size_t(group) * columns * 2,
                        tensor.data() +
                            (size_t(group) * vocabulary + head.gpuColumns) * 8,
                        size_t(columns) * 8);
          packed = subset.data();
          affine = parameters.data();
        }
        if (head.create(columns, 2048, packed, affine, &head.projector, error,
                        sizeof(error)))
          throw std::runtime_error(
              std::string("Hexagon head initialization: ") + error);
        std::vector<char>().swap(head.packed);
        head.input.resize(2048);
        head.logits.resize(columns);
        android_model_set_head_columns(state.get(), head.gpuColumns);
        std::printf("Inference routing: Mojo Adreno layers; vocabulary head "
                    "GPU=%u Hexagon=%u rows\n",
                    head.gpuColumns, columns);
      }
#endif
      if (centeredParams(i)) {
        std::vector<_Float16> scales(size / 8);
        for (uint64_t offset = 0; offset < size; offset += 8) {
          float scale;
          std::memcpy(&scale, tensor.data() + offset, 4);
          scales[offset / 8] = _Float16(scale);
          require(float(scales[offset / 8]) == scale,
                  "Packed scale must be exactly representable in FP16");
        }
        check(
            android_model_allocate(state.get(), 1, size / 4, scales.data(), 0));
        totalBytes += size / 4;
      } else {
        check(android_model_allocate(state.get(), 1, size, tensor.data(), 0));
        totalBytes += size;
      }
    }
    require(model.peek() == std::char_traits<char>::eof(),
            "Trailing model data");
    for (int i = 0; i < bufferCount; ++i) {
      if (i == ropeBuffer) {
        check(android_model_alias_weight(state.get(), weightCount - 1));
        continue;
      }
      uint64_t size = bufferElements(i, capacity) * 4;
      check(
          android_model_allocate(state.get(), 0, size, nullptr, zeroBuffer(i)));
      totalBytes += size;
    }
    std::printf("Device allocations: %.2f MiB; context capacity: %u\n",
                totalBytes / 1048576., capacity);
#ifdef BONSAI_MODEL
    greedyCandidates.resize(size_t(vocabulary / 512) * 3);
#endif
  }
};
AndroidLlmEngine::AndroidLlmEngine(const std::string &plugin,
                                   const std::string &weights)
    : impl(std::make_unique<Impl>(plugin, weights)) {}
AndroidLlmEngine::~AndroidLlmEngine() = default;
uint32_t AndroidLlmEngine::capacity() const { return impl->capacity; }
const std::string &AndroidLlmEngine::deviceName() const {
  return impl->deviceName;
}
void AndroidLlmEngine::synchronize() {
  check(android_model_synchronize(impl->state.get()));
}
void AndroidLlmEngine::reset() {
  synchronize();
  for (int i = 0; i < bufferCount; ++i)
    if (zeroBuffer(i))
      check(android_model_zero(impl->state.get(), i));
  impl->position = 0;
  if (impl->hexagon)
    impl->hexagon->ready = false;
}
void AndroidLlmEngine::forward(int32_t token) {
  if (impl->hexagon)
    impl->hexagon->ready = false;
  require(token >= 0 && token < vocabulary && impl->position >= 0 &&
              uint32_t(impl->position) < impl->capacity,
          "Token or position out of range");
  if (impl->hexagon) {
    auto &head = *impl->hexagon;
    auto start = std::chrono::steady_clock::now();
    check(android_model_forward_with_head_input(impl->state.get(), token,
                                                impl->position, impl->capacity,
                                                head.input.data()));
    auto submitted = std::chrono::steady_clock::now();
    char error[512]{};
    if (head.execute(head.projector, head.input.data(), head.logits.data(),
                     error, sizeof(error)))
      throw std::runtime_error(std::string("Hexagon head: ") + error);
    head.ready = true;
    if (std::getenv("MODULAR_HEXAGON_TIMING")) {
      auto done = std::chrono::steady_clock::now();
      std::fprintf(
          stderr, "XPU_TIMING position=%d layers_ms=%.3f rpc_ms=%.3f\n",
          impl->position,
          std::chrono::duration<double, std::milli>(submitted - start).count(),
          std::chrono::duration<double, std::milli>(done - submitted).count());
    }
  } else {
    check(android_model_forward(impl->state.get(), token, impl->position,
                                impl->capacity));
  }
  ++impl->position;
}
std::vector<float> AndroidLlmEngine::readLogits() {
  if (impl->hexagon && impl->hexagon->ready) {
    auto &head = *impl->hexagon;
    if (!head.gpuColumns)
      return head.logits;
    std::vector<float> result(vocabulary);
    check(android_model_read(impl->state.get(), logitsBuffer, result.data(),
                             head.gpuColumns * 4ull));
    std::copy(head.logits.begin(), head.logits.end(),
              result.begin() + head.gpuColumns);
    return result;
  }
  std::vector<float> logits(vocabulary);
  check(android_model_read(impl->state.get(), logitsBuffer, logits.data(),
                           logits.size() * sizeof(float)));
  return logits;
}
void AndroidLlmEngine::enqueueLogitsRead(float *destination) {
  require(!impl->hexagon, "Async full-logit transfer requires the GPU-only path");
  check(android_model_read_async(impl->state.get(), logitsBuffer, destination,
                                size_t(vocabulary) * sizeof(float)));
}
int AndroidLlmEngine::readGreedyToken() {
#ifdef BONSAI_MODEL
  if (impl->hexagon && impl->hexagon->ready) {
    auto logits = readLogits();
    return greedyToken(logits.data(), logits.size());
  }
  enqueueGreedyTokenRead();
  return finishGreedyTokenRead();
#else
  auto logits = readLogits();
  return greedyToken(logits.data(), logits.size());
#endif
}
void AndroidLlmEngine::enqueueGreedyTokenRead() {
#ifdef BONSAI_MODEL
  if (impl->hexagon && impl->hexagon->ready)
    throw std::runtime_error(
        "Async selection read is unavailable with Hexagon");
  if (impl->greedyReadPending)
    throw std::runtime_error("A greedy selection read is already pending");
  check(android_model_read_async(
      impl->state.get(), greedyBuffer, impl->greedyCandidates.data(),
      impl->greedyCandidates.size() * sizeof(float)));
  impl->greedyReadPending = true;
#else
  throw std::runtime_error("Async selection read is only available for Bonsai");
#endif
}
int AndroidLlmEngine::finishGreedyTokenRead() {
#ifdef BONSAI_MODEL
  if (!impl->greedyReadPending)
    throw std::runtime_error("No greedy selection read is pending");
  check(android_model_synchronize(impl->state.get()));
  impl->greedyReadPending = false;
  constexpr size_t candidateCount = vocabulary / 512;
  float best = -std::numeric_limits<float>::infinity();
  int bestIndex = -1;
  for (size_t i = 0; i < candidateCount; ++i) {
    float value = impl->greedyCandidates[i * 3];
    float index = impl->greedyCandidates[i * 3 + 1];
    float invalid = impl->greedyCandidates[i * 3 + 2];
    if (!std::isfinite(value) || invalid != 0 || !std::isfinite(index) ||
        index < 0 || index >= vocabulary || index != int(index))
      throw std::runtime_error("Empty or non-finite fused greedy candidates");
    if (value > best || (value == best && int(index) < bestIndex)) {
      best = value;
      bestIndex = int(index);
    }
  }
  if (bestIndex < 0)
    throw std::runtime_error("Empty fused greedy candidates");
  return bestIndex;
#else
  throw std::runtime_error("Async selection read is only available for Bonsai");
#endif
}
