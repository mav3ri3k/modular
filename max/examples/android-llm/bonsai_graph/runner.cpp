// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Checkpoint I/O and sampling only. MAX executes the compiled Mojo graph.
#include "../model_spec.h"
#include "../sampling.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/device.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/weights.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" void KGEN_CompilerRT_DestroyGlobals();
struct RuntimeGlobals {
  ~RuntimeGlobals() { KGEN_CompilerRT_DestroyGlobals(); }
};
template <auto Free>
struct Deleter {
  template <typename T>
  void operator()(T *p) const {
    Free(p);
  }
};
template <typename T, auto Free>
using Handle = std::unique_ptr<T, Deleter<Free>>;
using TensorMap = Handle<M_AsyncTensorMap, M_freeAsyncTensorMap>;
static void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
static void check(M_Status *status, const char *where) {
  if (M_isError(status))
    throw std::runtime_error(std::string(where) + ": " + M_getError(status));
}
static double seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}
struct Checkpoint {
  void *mapping = MAP_FAILED;
  size_t size = 0;
  std::vector<const void *> tensors;
  std::vector<std::vector<_Float16>> scales{weightCount};
  explicit Checkpoint(const char *path, uint32_t capacity) {
    int fd = open(path, O_RDONLY);
    require(fd >= 0, "Cannot open packed checkpoint");
    struct stat statbuf{};
    if (fstat(fd, &statbuf)) {
      close(fd);
      throw std::runtime_error("Cannot stat checkpoint");
    }
    size = size_t(statbuf.st_size);
    mapping = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    require(mapping != MAP_FAILED, "Cannot map checkpoint");
    // Validate before exposing any borrowed weight data to MAX.
    try {
      auto *bytes = static_cast<const unsigned char *>(mapping);
      size_t cursor = 0;
      auto take = [&](size_t n) {
        require(cursor <= size && n <= size - cursor, "Truncated checkpoint");
        auto *result = bytes + cursor;
        cursor += n;
        return result;
      };
      require(std::memcmp(take(8), modelMagic, 8) == 0, "Bad checkpoint magic");
      uint32_t count, actualCapacity;
      std::memcpy(&count, take(4), 4);
      std::memcpy(&actualCapacity, take(4), 4);
      require(count == weightCount && actualCapacity == capacity,
              "Checkpoint count/capacity differs from graph");
      std::vector<const void *> shared(48, nullptr);
      for (int i = 0; i < weightCount; ++i) {
        uint64_t length;
        std::memcpy(&length, take(8), 8);
        require(length == weightBytes(i, capacity), "Bad weight size");
        auto *data = take(length);
        tensors.push_back(data);
        if (centeredParams(i)) {
          scales[i].resize(length / 8);
          for (uint64_t offset = 0; offset < length; offset += 8) {
            float scale, bias;
            std::memcpy(&scale, data + offset, 4);
            std::memcpy(&bias, data + offset + 4, 4);
            require(std::isfinite(scale) && bias == -scale,
                    "Packed groups must be finite and centered");
            scales[i][offset / 8] = _Float16(scale);
            require(float(scales[i][offset / 8]) == scale,
                    "Packed scale must be exactly representable in FP16");
          }
          tensors.back() = scales[i].data();
        }
        int group = sharedSignGroup(i);
        if (group >= 0) {
          if (shared[group])
            require(std::memcmp(shared[group], data, length) == 0,
                    "Shared projection signs differ");
          else
            shared[group] = data;
        }
      }
      require(cursor == size, "Unexpected checkpoint trailing bytes");
    } catch (...) {
      munmap(mapping, size);
      mapping = MAP_FAILED;
      throw;
    }
  }
  ~Checkpoint() {
    if (mapping != MAP_FAILED)
      munmap(mapping, size);
  }
};

int main(int argc, char **argv) {
  RuntimeGlobals globals;
  try {
    require(argc == 8, "usage: graph_runner model.mef weights.bin prompt.txt "
                       "reference.f32-or-benchmark steps layers capacity");
    int steps = std::stoi(argv[5]), layers = std::stoi(argv[6]);
    int capacity = std::stoi(argv[7]);
    require(steps > 0 && layers > 0 && layers <= 24 && capacity > 0 &&
                capacity <= 8192,
            "Bad run dimensions");
    Handle<M_Status, M_freeStatus> status(M_newStatus());
    Handle<M_RuntimeConfig, M_freeRuntimeConfig> config(M_newRuntimeConfig());
    Handle<M_Device, M_freeDevice> cpu(M_newDevice(M_HOST, 0, status.get()));
    check(status.get(), "CPU device");
    Handle<M_Device, M_freeDevice> gpu(
        M_newDevice(M_ACCELERATOR, 0, status.get()));
    check(status.get(), "OpenCL device");
    M_runtimeConfigAddDevice(config.get(), cpu.get());
    M_runtimeConfigAddDevice(config.get(), gpu.get());
    Handle<M_RuntimeContext, M_freeRuntimeContext> context(
        M_newRuntimeContext(config.get(), status.get()));
    check(status.get(), "MAX context");
    Checkpoint checkpoint(argv[2], capacity);
    std::vector<std::string> names;
    for (int i = 0; i < weightCount; ++i) {
      char name[16];
      std::snprintf(name, sizeof(name), "w%03d", i);
      names.emplace_back(name);
    }
    std::vector<const char *> namePointers;
    for (auto &name : names)
      namePointers.push_back(name.c_str());
    Handle<M_WeightsRegistry, M_freeWeightsRegistry> weights(
        M_newWeightsRegistry(namePointers.data(), checkpoint.tensors.data(),
                             weightCount, status.get()));
    check(status.get(), "Weights registry");
    M_CompileConfig *compileConfig = M_newCompileConfig();
    M_setModelPath(compileConfig, argv[1]);
    Handle<M_AsyncCompiledModel, M_freeCompiledModel> compiled(
        M_compileModelSync(context.get(), &compileConfig, status.get()));
    check(status.get(), "MEF load");
    Handle<M_AsyncModel, M_freeModel> model(M_initModel(
        context.get(), compiled.get(), weights.get(), status.get()));
    check(status.get(), "MAX model init");
    M_waitForModel(model.get(), status.get());
    check(status.get(), "MAX model ready");
    TensorMap inputs(M_newAsyncTensorMap(context.get()));
    auto borrow = [&](int index, void *data, int64_t n, M_Dtype dtype,
                      M_Device *device) {
      std::string name = "input" + std::to_string(index);
      Handle<M_TensorSpec, M_freeTensorSpec> spec(
          M_newTensorSpec(&n, 1, dtype, name.c_str(), device));
      M_borrowTensorInto(inputs.get(), data, spec.get(), status.get());
      check(status.get(), "Persistent graph input");
    };
    int32_t token = 0, position = 0;
    borrow(0, &token, 1, M_INT32, cpu.get());
    borrow(1, &position, 1, M_INT32, cpu.get());
    for (int i = 0; i < layers * 2; ++i) {
      std::vector<float> zero(bufferElements(19 + i, capacity), 0);
      // Host data is staged once; the map owns the GPU allocation thereafter.
      borrow(i + 2, zero.data(), zero.size(), M_FLOAT32, gpu.get());
    }
    std::ifstream prompt(argv[3]);
    std::vector<int32_t> tokens;
    while (prompt >> token) {
      require(token >= 0 && token < vocabulary, "Bad prompt token");
      tokens.push_back(token);
    }
    require(!tokens.empty() && tokens.size() + steps <= size_t(capacity),
            "Prompt exceeds graph context");
    bool validate = std::strcmp(argv[4], "benchmark") != 0;
    std::ifstream reference(argv[4], std::ios::binary);
    require(!validate || bool(reference), "Missing logit reference");
    auto forward = [&](int32_t next) {
      require(next >= 0 && next < vocabulary && position < capacity,
              "Invalid graph input");
      token = next;
      TensorMap outputs(M_executeModelSync(context.get(), model.get(),
                                           inputs.get(), status.get()));
      check(status.get(), "MAX execute");
      ++position;
      return outputs;
    };
    auto start = std::chrono::steady_clock::now();
    TensorMap outputs;
    for (int32_t next : tokens)
      outputs = forward(next);
    std::printf("Prefill: %zu tokens, %.3f s\n", tokens.size(), seconds(start));
    start = std::chrono::steady_clock::now();
    double maxError = 0, sumSquare = 0, refSquare = 0;
    std::vector<float> expected(vocabulary);
    for (int step = 0; step < steps; ++step) {
      Handle<M_AsyncTensor, M_freeTensor> result(
          M_getTensorByNameFrom(outputs.get(), "output0", status.get()));
      check(status.get(), "Graph logits");
      require(M_getTensorNumElements(result.get()) == vocabulary,
              "Wrong vocabulary size");
      auto *logits = static_cast<const float *>(M_getTensorData(result.get()));
      require(logits != nullptr, "No CPU logits");
      int next = greedyToken(logits, vocabulary);
      if (validate) {
        require(bool(reference.read(reinterpret_cast<char *>(expected.data()),
                                    expected.size() * sizeof(float))),
                "Truncated reference logits");
        for (int i = 0; i < vocabulary; ++i) {
          require(std::isfinite(expected[i]), "Non-finite reference logits");
          double delta = double(logits[i]) - expected[i];
          maxError = std::max(maxError, std::abs(delta));
          sumSquare += delta * delta;
          refSquare += double(expected[i]) * expected[i];
        }
      }
      std::printf("TOKEN %d %d\n", step, next);
      std::fflush(stdout);
      if (validate)
        require(next == std::max_element(expected.begin(), expected.end()) -
                            expected.begin(),
                "Graph greedy ID differs from reference");
      if (step + 1 < steps)
        outputs = forward(next);
    }
    double decode = seconds(start);
    std::printf("Decode: %d forwards, %.3f s, %.2f tok/s (MAX graph, includes "
                "logit readback%s)\n",
                steps - 1, decode, (steps - 1) / decode,
                validate ? " and validation" : "");
    if (validate) {
      double relative = std::sqrt(sumSquare / std::max(refSquare, 1e-30));
      std::printf("Logits: max_abs_error=%.6g relative_l2=%.6g\n", maxError,
                  relative);
      require(maxError < .05 && relative < .005,
              "Graph logit tolerance exceeded");
      std::puts("PASS: all generated tokens and logits match reference");
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
