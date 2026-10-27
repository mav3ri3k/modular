// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// Owns a loaded Mojo model and DeviceContext. Call on one worker thread.
class AndroidLlmEngine {
public:
  AndroidLlmEngine(const std::string &plugin, const std::string &weights);
  ~AndroidLlmEngine();
  AndroidLlmEngine(const AndroidLlmEngine &) = delete;
  AndroidLlmEngine &operator=(const AndroidLlmEngine &) = delete;
  uint32_t capacity() const;
  const std::string &deviceName() const;
  /// Reset caches and position, retaining all model weights and kernels.
  void reset();
  /// Enqueue one token at the next position. Throws on HAL errors.
  void forward(int32_t token);
  void synchronize();
  /// Read the last token's vocabulary logits, waiting for queued work.
  std::vector<float> readLogits();
  /// Enqueue all vocabulary logits into caller storage, valid until synchronize.
  /// Used to match llama_decode output transfer without allocation or sampling.
  void enqueueLogitsRead(float *destination);
  /// Read the fused head's greedy token ID, waiting for queued work.
  int readGreedyToken();
  /// Enqueue the fused candidates read. Keep the destination live until finish.
  void enqueueGreedyTokenRead();
  /// Complete a queued fused candidates read and select the token.
  int finishGreedyTokenRead();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
