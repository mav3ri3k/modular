// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#include "engine.h"
#include "model_spec.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>
extern "C" void KGEN_CompilerRT_DestroyGlobals();
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      return 2;
    std::vector<float> logits(vocabulary);
    {
      AndroidLlmEngine e(argv[1], argv[2]);
      std::srand(1);
      auto step = [&]() {
        e.forward(std::rand() % vocabulary);
        e.enqueueLogitsRead(logits.data());
        e.synchronize();
      };
      step();
      e.reset();
      e.synchronize();
      for (int i = 0; i < std::atoi(argv[3]); i++) {
        auto start = std::chrono::steady_clock::now();
        step();
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
        std::printf("TOKEN_MS %d %.6f\n", i, ms);
        std::fflush(stdout);
      }
    }
    KGEN_CompilerRT_DestroyGlobals();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR %s\n", e.what());
    return 1;
  }
}
