// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#pragma once
#include <cstdint>
#if defined(QWEN_MODEL)
#include "qwen_spec.h"
#elif defined(BONSAI_MODEL)
static constexpr char modelMagic[] = "BONSAI01";
static constexpr int vocabulary = 248320, weightCount = 605, bufferCount = 69;
// Stop IDs come from the checkpoint generation_config.json (chat and text EOS).
static constexpr int logitsBuffer = 17, ropeBuffer = 18, greedyBuffer = 68,
                     eosToken = 248046, alternateEosToken = 248044;
// Sign vectors may only be shared within these verified projection groups.
static inline int sharedSignGroup(int i) {
  if (i < 4 || i >= 604)
    return -1;
  int layer = (i - 4) / 25, slot = (i - 4) % 25;
  if (slot == 4 || slot == 7)
    return layer * 2;
  if (slot == 13 || slot == 16 || (slot == 19 && (layer + 1) % 4 == 0))
    return layer * 2 + 1;
  return -1;
}
// The fast centered-dot kernel relies on this checkpoint contract. Validate
// these scale/bias tensors once during model loading, before any dispatch.
static inline bool centeredParams(int i) {
  if (i == 1)
    return true;
  if (i < 4 || i >= 604)
    return false;
  int layer = (i - 4) / 25, slot = (i - 4) % 25;
  return slot == 3 || slot == 6 || slot == 9 || slot == 12 || slot == 15 ||
         slot == 18 || ((layer + 1) % 4 == 0 && slot == 21);
}
static inline uint64_t packedBytes(int part, uint64_t out, uint64_t in) {
  return part == 0   ? out * in / 16 * 4
         : part == 1 ? out * in / 128 * 2 * 4
                     : in * 4;
}
static inline uint64_t weightBytes(int i, uint32_t capacity) {
  if (i < 3)
    return packedBytes(i, 248320, 2048);
  if (i == 3)
    return 2048 * 4;
  if (i == 604)
    return capacity * 64ull * 4;
  int layer = (i - 4) / 25, slot = (i - 4) % 25;
  if (slot < 2)
    return 2048 * 4;
  if (slot < 11) {
    int group = (slot - 2) / 3, part = (slot - 2) % 3;
    return packedBytes(part, group == 2 ? 2048 : 6144,
                       group == 2 ? 6144 : 2048);
  }
  if ((layer + 1) % 4 == 0) {
    if (slot >= 23)
      return 256 * 4;
    constexpr int width[] = {4096, 512, 512, 2048};
    return packedBytes((slot - 11) % 3, width[(slot - 11) / 3], 2048);
  }
  if (slot < 20)
    return packedBytes((slot - 11) % 3, slot < 14 ? 6144 : 2048, 2048);
  if (slot < 22)
    return 2048 * 16 * 4;
  if (slot == 22)
    return 32 * 4;
  if (slot == 23)
    return 6144 * 4 * 4;
  return 128 * 4;
}
static inline uint64_t bufferElements(int i, uint32_t capacity) {
  if (i == 68)
    return 1455; // 485 subgroup candidates, each value/index/invalid.
  if (i == 67)
    return 12288; // Eight-group join: 6144*2 or 2048*6 partials.
  constexpr uint64_t scratch[] = {2048, 2048, 6144, 6144, 6144, 4096,
                                  16,   16,   32,   6144, 2048, 6144,
                                  6144, 0,    2048, 512,  512,  248320};
  if (i < 18)
    return i == 13 ? 8ull * capacity : scratch[i];
  int layer = (i - 19) / 2, part = (i - 19) % 2;
  return (layer + 1) % 4 == 0 ? 512ull * capacity
         : part == 0          ? 3ull * 6144
                              : 16ull * 128 * 128;
}
static inline bool zeroBuffer(int i) {
  return i >= 19 && i < 67 && (((i - 19) / 2 + 1) % 4 != 0);
}
#else
static inline int sharedSignGroup(int) { return -1; }
static inline bool centeredParams(int) { return false; }
static constexpr char modelMagic[] = "SMOLBF01";
static constexpr int vocabulary = 49152, weightCount = 274, bufferCount = 13;
static constexpr int logitsBuffer = 9, ropeBuffer = 12, eosToken = 2,
                     alternateEosToken = 2;
static inline uint64_t weightBytes(int i, uint32_t capacity) {
  if (i < 2)
    return 49152ull * 576 * 2;
  if (i == 2)
    return 576 * 2;
  if (i == 273)
    return capacity * 64ull * 4;
  constexpr uint64_t sizes[] = {576,        576 * 576,  576 * 192,
                                576 * 192,  576 * 576,  576,
                                576 * 1536, 576 * 1536, 1536 * 576};
  return sizes[(i - 3) % 9] * 2;
}
static inline uint64_t bufferElements(int i, uint32_t capacity) {
  const uint64_t elements[] = {576,
                               576,
                               576,
                               192,
                               192,
                               576,
                               9ull * capacity,
                               1536,
                               1536,
                               49152,
                               30ull * capacity * 192,
                               30ull * capacity * 192};
  return elements[i];
}
static inline bool zeroBuffer(int) { return false; }
#endif
