// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt Link against pinned Prism llama.cpp, independent
// teacher-forced reference.
#include "ggml-backend.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
int main(int argc, char **argv) {
  try {
    if (argc < 5 || argc > 7)
      throw std::runtime_error("usage: reference gguf token_ids output.f32 "
                               "gpu_layers [f16|f32] [capacity]");
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = std::stoi(argv[4]);
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model)
      throw std::runtime_error("load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = argc == 7 ? std::stoi(argv[6]) : 1024;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = argc >= 6 && std::strcmp(argv[5], "f16") == 0 ? GGML_TYPE_F16
                                                              : GGML_TYPE_F32;
    cp.type_v = cp.type_k;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx)
      throw std::runtime_error("context failed");
    std::ifstream tokens(argv[2]);
    std::ofstream out(argv[3], std::ios::binary);
    int id, n = 0;
    if (!tokens || !out)
      throw std::runtime_error("input/output failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    while (tokens >> id) {
      llama_token token = id;
      if (llama_decode(ctx, llama_batch_get_one(&token, 1)))
        throw std::runtime_error("decode failed");
      llama_synchronize(ctx);
      auto *logits = llama_get_logits_ith(ctx, -1);
      out.write(reinterpret_cast<const char *>(logits), vocab * 4);
      ++n;
    }
    if (!tokens.eof() || !n || !out)
      throw std::runtime_error("bad token/output");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::printf("Dumped %d reference logits\n", n);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
