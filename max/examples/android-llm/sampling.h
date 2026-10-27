// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

// Independent reductions let the host compiler vectorize the scan. Keep the
// first index on ties, including ties between lanes, and reject nonfinite
// logits.
inline int greedyToken(const float *logits, size_t count) {
  constexpr size_t lanes = 8;
  float best[lanes];
  size_t indices[lanes] = {};
  for (size_t lane = 0; lane < lanes; ++lane)
    best[lane] = -std::numeric_limits<float>::infinity();
  unsigned invalid = 0;
  size_t i = 0;
  for (; i + lanes <= count; i += lanes) {
    for (size_t lane = 0; lane < lanes; ++lane) {
      float value = logits[i + lane];
      invalid |= !std::isfinite(value);
      bool greater = value > best[lane];
      best[lane] = greater ? value : best[lane];
      indices[lane] = greater ? i + lane : indices[lane];
    }
  }
  float maximum = best[0];
  size_t index = indices[0];
  for (size_t lane = 1; lane < lanes; ++lane) {
    if (best[lane] > maximum ||
        (best[lane] == maximum && indices[lane] < index)) {
      maximum = best[lane];
      index = indices[lane];
    }
  }
  for (; i < count; ++i) {
    float value = logits[i];
    invalid |= !std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      index = i;
    }
  }
  if (invalid || !count)
    throw std::runtime_error("Empty or non-finite logits");
  return int(index);
}
