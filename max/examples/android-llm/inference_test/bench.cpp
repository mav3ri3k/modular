// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt llama-bench test_gen semantics: random input,
// one forward and sync per token, full-logit readback, no sampling, one-token warmup, clear
// cache outside measured interval.
#include "engine.h"
#include "model_spec.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
extern "C" void KGEN_CompilerRT_DestroyGlobals();
int main(int argc, char **argv) {
  struct Globals {
    ~Globals() { KGEN_CompilerRT_DestroyGlobals(); }
  } globals;
  try {
    if (argc != 8)
      throw std::runtime_error("usage: bench plugin weights n_prompt n_gen "
                               "n_depth repetitions seed");
    int pp = std::stoi(argv[3]), tg = std::stoi(argv[4]),
        depth = std::stoi(argv[5]);
    int reps = std::stoi(argv[6]), seed = std::stoi(argv[7]);
    // Keep transfer storage alive through engine destruction on exceptional exits.
    std::vector<float> logits(vocabulary);
    AndroidLlmEngine engine(argv[1], argv[2]);
    if (pp < 0 || tg < 0 || depth < 0 || reps < 1 || pp + tg < 1 ||
        uint64_t(pp) + tg + depth > engine.capacity())
      throw std::runtime_error("invalid workload/capacity");
    std::srand(seed);
    auto prompt = [&](int n) {
      for (int i = 0; i < n; ++i) {
        engine.forward(std::rand() % vocabulary);
        engine.enqueueLogitsRead(logits.data());
      }
      engine.synchronize();
    };
    auto gen = [&](int n) {
      int token = std::rand() % vocabulary;
      for (int i = 0; i < n; ++i) {
        engine.forward(token);
        engine.enqueueLogitsRead(logits.data());
        engine.synchronize();
        token = std::rand() % vocabulary;
      }
    };
    if (pp)
      prompt(pp);
    if (tg)
      gen(1);
    std::vector<uint64_t> ns;
    for (int r = 0; r < reps; ++r) {
      engine.reset();
      engine.synchronize(); // wait for recurrent state clears
      if (depth)
        prompt(depth);
      auto start = std::chrono::steady_clock::now();
      if (pp)
        prompt(pp);
      if (tg)
        gen(tg);
      ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count());
    }
    double avg = 0, dev = 0;
    for (auto t : ns)
      avg += 1e9 * (pp + tg) / t;
    avg /= reps;
    for (auto t : ns)
      dev += std::pow(1e9 * (pp + tg) / t - avg, 2);
    dev = reps > 1 ? std::sqrt(dev / (reps - 1)) : 0;
    std::printf("RESULT "
                "{\"n_prompt\":%d,\"n_gen\":%d,\"n_depth\":%d,\"n_batch\":1,"
                "\"n_ubatch\":1,\"capacity\":%u,\"seed\":%d,\"avg_ts\":%.9f,"
                "\"stddev_ts\":%.9f,\"logit_readback\":true,\"samples_ns\":[",
                pp, tg, depth, engine.capacity(), seed, avg, dev);
    for (size_t i = 0; i < ns.size(); ++i)
      std::printf("%s%llu", i ? "," : "", (unsigned long long)ns[i]);
    std::puts("]}");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
