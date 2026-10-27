// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "Target/HexagonProjection.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <vector>
template <class T>
T read(std::ifstream &f) {
  T v{};
  if (!f.read(reinterpret_cast<char *>(&v), sizeof(v)))
    throw std::runtime_error("Truncated checkpoint");
  return v;
}
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error("usage: projection_bench HexagonProjection.so "
                               "weights.bin output_width");
    unsigned width = std::stoul(argv[3]), k = 2048, full = 248320;
    if (!width || width > full || width % 32)
      throw std::runtime_error("Invalid width");
    std::ifstream f(argv[2], std::ios::binary);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "BONSAI01", 8) || read<unsigned>(f) != 605)
      throw std::runtime_error("Expected Bonsai export");
    read<unsigned>(f);
    if (read<uint64_t>(f) != uint64_t(full) * k / 4)
      throw std::runtime_error("Bad packed head size");
    std::vector<uint32_t> packed(size_t(width) * k / 16);
    for (unsigned word = 0; word < k / 16; ++word) {
      f.read(reinterpret_cast<char *>(packed.data() + size_t(word) * width),
             width * 4);
      f.seekg((full - width) * 4, std::ios::cur);
    }
    if (read<uint64_t>(f) != uint64_t(full) * k / 128 * 8)
      throw std::runtime_error("Bad scale size");
    std::vector<float> affine(size_t(width) * k / 128 * 2);
    for (unsigned group = 0; group < k / 128; ++group) {
      f.read(
          reinterpret_cast<char *>(affine.data() + size_t(group) * width * 2),
          width * 8);
      f.seekg((full - width) * 8, std::ios::cur);
    }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib)
      throw std::runtime_error(dlerror());
    auto create = reinterpret_cast<decltype(&M_hexagon_projection_create)>(
        dlsym(lib, "M_hexagon_projection_create"));
    auto execute = reinterpret_cast<decltype(&M_hexagon_projection_execute)>(
        dlsym(lib, "M_hexagon_projection_execute"));
    auto destroy = reinterpret_cast<decltype(&M_hexagon_projection_destroy)>(
        dlsym(lib, "M_hexagon_projection_destroy"));
    if (!create || !execute || !destroy)
      throw std::runtime_error("Missing projector API");
    char error[512]{};
    M_hexagon_projection_t handle = nullptr;
    if (create(width, k, packed.data(), affine.data(), &handle, error,
               sizeof(error)))
      throw std::runtime_error(error);
    std::vector<uint16_t> x(k);
    std::vector<float> y(width);
    for (unsigned i = 0; i < k; ++i) {
      __fp16 h = __fp16((int(i % 31) - 15) / 16.0f);
      std::memcpy(&x[i], &h, 2);
    }
    std::vector<double> ms;
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto start = std::chrono::steady_clock::now();
      if (execute(handle, x.data(), y.data(), error, sizeof(error))) {
        destroy(handle);
        throw std::runtime_error(error);
      }
      ms.push_back(std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                       .count());
    }
    double maxError = 0;
    for (unsigned out = 0; out < width; ++out) {
      float expected = 0;
      for (unsigned group = 0; group < k / 128; ++group) {
        float dot = 0;
        for (unsigned lane = 0; lane < 128; ++lane) {
          unsigned i = group * 128 + lane;
          __fp16 h;
          std::memcpy(&h, &x[i], 2);
          unsigned c =
              (packed[size_t(i / 16) * width + out] >> ((i % 16) * 2)) & 3;
          dot += float(h) * (int(c) - 1);
        }
        expected += dot * affine[(size_t(group) * width + out) * 2];
      }
      if (!std::isfinite(y[out]))
        throw std::runtime_error("Non-finite Hexagon result");
      maxError = std::max(maxError, double(std::abs(expected - y[out])));
    }
    destroy(handle);
    dlclose(lib);
    std::printf("Mojo Hexagon Bonsai projection %ux%u: ms=%.3f,%.3f,%.3f "
                "max_abs_error=%.8g\n",
                width, k, ms[0], ms[1], ms[2], maxError);
    if (maxError > 0.02)
      throw std::runtime_error("Projection does not match CPU reference");
    puts("PASS: real packed Bonsai weights executed on Hexagon");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
