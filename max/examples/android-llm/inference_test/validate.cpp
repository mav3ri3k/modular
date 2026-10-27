// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#include "engine.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
extern "C" void KGEN_CompilerRT_DestroyGlobals();
int main(int argc, char **argv) {
  struct Globals {
    ~Globals() { KGEN_CompilerRT_DestroyGlobals(); }
  } globals;
  try {
    if (argc != 5)
      throw std::runtime_error(
          "usage: validate plugin weights token_ids.txt output.f32");
    AndroidLlmEngine engine(argv[1], argv[2]);
    std::ifstream tokens(argv[3]);
    std::ofstream out(argv[4], std::ios::binary);
    if (!tokens || !out)
      throw std::runtime_error("cannot open input/output");
    int id, n = 0;
    while (tokens >> id) {
      engine.forward(id);
      auto logits = engine.readLogits();
      out.write(reinterpret_cast<const char *>(logits.data()),
                logits.size() * 4);
      ++n;
    }
    if (!tokens.eof() || n == 0 || !out)
      throw std::runtime_error("bad tokens/output");
    std::printf("Dumped %d teacher-forced vocabulary logits\n", n);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
