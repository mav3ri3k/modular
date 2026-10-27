// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Transport, checkpoint I/O and greedy sampling only. Forward pass is in Mojo.
#include "engine.h"
#include "model_spec.h"
#include "sampling.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void KGEN_CompilerRT_DestroyGlobals();
struct RuntimeGlobals {
  ~RuntimeGlobals() { KGEN_CompilerRT_DestroyGlobals(); }
};

static void require(bool value, const char *what) {
  if (!value)
    throw std::runtime_error(what);
}
static double seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}
int main(int argc, char **argv) {
  RuntimeGlobals runtimeGlobals;
  try {
    require(argc == 6,
            "usage: runner OpenCLPlugin.so weights.bin "
            "prompt.txt reference.f32-or-dash-or-benchmark generation_steps");
    int steps = std::stoi(argv[5]);
    require(steps >= 1 && steps <= 8192, "Invalid generation length");
    AndroidLlmEngine engine(argv[1], argv[2]);
    const uint32_t capacity = engine.capacity();
    std::printf("GPU: %s\n", engine.deviceName().c_str());
    std::ifstream promptFile(argv[3]);
    std::vector<int32_t> tokens;
    int32_t token;
    while (promptFile >> token) {
      require(token >= 0 && token < vocabulary, "Bad prompt token");
      tokens.push_back(token);
    }
    require(!tokens.empty() && tokens.size() + steps <= capacity,
            "Invalid prompt or context overflow");
    std::ifstream reference(argv[4], std::ios::binary);
    bool benchmark = std::strcmp(argv[4], "benchmark") == 0;
    bool validate = !benchmark && std::strcmp(argv[4], "-") != 0;
    require(!validate || bool(reference), "Missing reference logits");
    auto forward = [&](int32_t id) { engine.forward(id); };
    std::vector<float> logits(vocabulary), expected(vocabulary);
    auto start = std::chrono::steady_clock::now();
    for (auto id : tokens)
      forward(id);
    engine.synchronize();
    double prefill = seconds(start);
    std::printf("Prefill: %zu tokens, %.3f s, %.2f tok/s\n", tokens.size(),
                prefill, tokens.size() / prefill);
    start = std::chrono::steady_clock::now();
    double maxError = 0, sumSquare = 0, refSquare = 0;
#ifdef BONSAI_MODEL
    bool asyncGreedy = benchmark && std::getenv("MODULAR_ASYNC_GREEDY");
#else
    bool asyncGreedy = false;
#endif
    if (asyncGreedy)
      engine.enqueueGreedyTokenRead();
    for (int step = 0; step < steps; ++step) {
      int next = asyncGreedy ? engine.finishGreedyTokenRead()
                             : engine.readGreedyToken();
      if (validate) {
        logits = engine.readLogits();
        require(bool(reference.read(reinterpret_cast<char *>(expected.data()),
                                    expected.size() * 4)),
                "Truncated reference");
        for (size_t i = 0; i < logits.size(); ++i) {
          require(std::isfinite(logits[i]) && std::isfinite(expected[i]),
                  "Non-finite logits");
          double delta = double(logits[i]) - expected[i];
          maxError = std::max(maxError, std::abs(delta));
          sumSquare += delta * delta;
          refSquare += double(expected[i]) * expected[i];
        }
        require(next == greedyToken(logits.data(), logits.size()),
                "Fused greedy token differs from full-logit selection");
      }
      if (validate) {
        int wanted = int(std::max_element(expected.begin(), expected.end()) -
                         expected.begin());
        require(next == wanted, "Greedy token differs from reference");
      }
      bool stop = !validate && !benchmark &&
                  (next == eosToken || next == alternateEosToken);
      if (stop) {
        steps = step + 1;
      } else if (step + 1 < steps && asyncGreedy) {
        forward(next);
        engine.enqueueGreedyTokenRead();
      }
      std::printf("TOKEN %d %d\n", step, next);
      std::fflush(stdout);
      if (stop)
        break;
      if (step + 1 < steps && !asyncGreedy) {
        forward(next);
      }
    }
    double decode = seconds(start);
    double relative = std::sqrt(sumSquare / std::max(refSquare, 1e-30));
    std::printf("Decode: %d forwards, %.3f s, %.2f tok/s (includes token "
                "selection readback%s)\n",
                steps - 1, decode, (steps - 1) / decode,
                validate ? ", full-logit readback and validation" : "");
    if (validate) {
      std::printf("Logits: max_abs_error=%.6g relative_l2=%.6g\n", maxError,
                  relative);
      require(maxError < 0.05 && relative < 0.005,
              "Logits exceed reference validation tolerance");
      std::puts("PASS: all generated tokens and logits match reference");
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
