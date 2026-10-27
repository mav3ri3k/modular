// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#ifndef TARGET_OPENCL_BUNDLE_H
#define TARGET_OPENCL_BUNDLE_H
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace M::OpenCL {
// ABI v3: typed logical packets with opaque buffer-view leaves.
struct ArgumentABI {
  enum Kind : uint32_t { Scalar = 2, BufferView = 3, Aggregate = 4 };
  uint32_t kind;
  uint32_t size;
  struct Field {
    uint32_t offset;
    uint32_t kind;
    uint32_t size;
  };
  // Aggregate transport: aligned original payload followed by a sparse view
  // sidecar. A pointer at payload offset N uses the 24-byte slot at payload
  // size + 3*N. Pointer fields are eight-byte aligned; no pointer is serialized
  // as a native address. Total transport size is four times aligned payload.
  std::vector<Field> fields{};
};
struct KernelABI {
  std::string name;
  std::vector<ArgumentABI> arguments;
  std::string nativeName{};
  bool dynamicShared = false;
  uint32_t subgroupSize = 0;
  bool deviceErrors = false;
};
struct Bundle {
  std::vector<KernelABI> kernels;
  std::string source;
  static constexpr uint32_t version = 3;
  std::vector<std::string> requiredExtensions{};
  uint32_t openclCVersion = 120;
};
inline void appendU32(std::string &bytes, uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    bytes.push_back(char(value >> (i * 8)));
}
inline std::string encodeBundle(const Bundle &bundle) {
  std::string bytes = "MOJOCL03";

  appendU32(bytes, bundle.openclCVersion);
  appendU32(bytes, uint32_t(bundle.requiredExtensions.size()));
  for (auto &extension : bundle.requiredExtensions) {
    appendU32(bytes, uint32_t(extension.size()));
    bytes += extension;
  }

  appendU32(bytes, uint32_t(bundle.kernels.size()));
  for (const auto &kernel : bundle.kernels) {
    appendU32(bytes, uint32_t(kernel.name.size()));
    bytes += kernel.name;

    appendU32(bytes, uint32_t(kernel.nativeName.size()));
    bytes += kernel.nativeName;
    appendU32(bytes, (kernel.dynamicShared ? 1u : 0u) |
                         (kernel.deviceErrors ? 2u : 0u));
    appendU32(bytes, kernel.subgroupSize);

    appendU32(bytes, uint32_t(kernel.arguments.size()));
    for (auto arg : kernel.arguments) {
      appendU32(bytes, arg.kind);
      appendU32(bytes, arg.size);
      if (arg.kind == ArgumentABI::Aggregate) {
        appendU32(bytes, uint32_t(arg.fields.size()));
        for (auto field : arg.fields) {
          appendU32(bytes, field.offset);
          appendU32(bytes, field.kind);
          appendU32(bytes, field.size);
        }
      }
    }
  }
  appendU32(bytes, uint32_t(bundle.source.size()));
  bytes += bundle.source;
  return bytes;
}
inline bool decodeBundle(const char *data, size_t size, Bundle &output) {
  if (!data || size < 12)
    return false;
  const std::string magic(data, 8);
  if (magic != "MOJOCL03")
    return false;
  size_t cursor = 8;
  auto u32 = [&](uint32_t &value) {
    if (cursor > size || size - cursor < 4)
      return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i)
      value |= uint32_t(uint8_t(data[cursor++])) << (8 * i);
    return true;
  };
  auto text = [&](std::string &value, uint32_t count) {
    if (cursor > size || count > size - cursor)
      return false;
    value.assign(data + cursor, count);
    cursor += count;
    return true;
  };
  Bundle parsed;
  uint32_t count;

  if (!u32(parsed.openclCVersion) ||
      (parsed.openclCVersion != 120 && parsed.openclCVersion != 300))
    return false;
  if (!u32(count) || count > 64)
    return false;
  for (uint32_t i = 0; i < count; ++i) {
    std::string extension;
    uint32_t length;
    if (!u32(length) || !length || length > 256 || !text(extension, length))
      return false;
    for (char ch : extension)
      if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_'))
        return false;
    parsed.requiredExtensions.push_back(std::move(extension));
  }

  if (!u32(count) || !count || count > 1024)
    return false;
  for (uint32_t i = 0; i < count; ++i) {
    KernelABI kernel;
    uint32_t length, args;
    // Generic Mojo kernel names include their full specialized type signature.
    // Bound the string by the remaining bundle, not an identifier-size limit.
    if (!u32(length) || !length || !text(kernel.name, length) ||
        kernel.name.find('\0') != std::string::npos)
      return false;

    if (!u32(length) || !length || length > 4096 ||
        !text(kernel.nativeName, length))
      return false;
    for (size_t c = 0; c < kernel.nativeName.size(); ++c) {
      char ch = kernel.nativeName[c];
      bool alpha =
          (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
      if (!alpha && !(c && ch >= '0' && ch <= '9'))
        return false;
    }
    uint32_t flags;
    if (!u32(flags) || (flags & ~3u))
      return false;
    kernel.dynamicShared = (flags & 1u) != 0;
    kernel.deviceErrors = (flags & 2u) != 0;
    if (!u32(kernel.subgroupSize) ||
        (kernel.subgroupSize != 0 && kernel.subgroupSize != 64))
      return false;

    if (!u32(args) || args > 256)
      return false;
    for (const auto &previous : parsed.kernels)
      if (previous.name == kernel.name ||
          previous.nativeName == kernel.nativeName)
        return false;
    for (uint32_t j = 0; j < args; ++j) {
      ArgumentABI arg;
      if (!u32(arg.kind) || !u32(arg.size))
        return false;
      if (arg.kind == ArgumentABI::Aggregate) {
        uint32_t fieldCount;
        if (!arg.size || arg.size > 262144 || arg.size % 32 ||
            !u32(fieldCount) || !fieldCount || fieldCount > 256)
          return false;
        uint32_t end = 0;
        for (uint32_t f = 0; f < fieldCount; ++f) {
          ArgumentABI::Field field;
          if (!u32(field.offset) || !u32(field.kind) || !u32(field.size))
            return false;
          bool pointer = field.kind == ArgumentABI::BufferView;
          uint32_t width = pointer ? 8 : field.size;
          if ((!pointer && field.kind != ArgumentABI::Scalar) ||
              (pointer && (field.size != 24 || field.offset % 8)) ||
              (!pointer &&
               !(width == 1 || width == 2 || width == 4 || width == 8 ||
                 width == 16 || width == 32 || width == 64 || width == 128)) ||
              field.offset < end || field.offset > arg.size / 4 ||
              width > arg.size / 4 - field.offset)
            return false;
          end = field.offset + width;
          arg.fields.push_back(field);
        }
      } else
        return false;
      kernel.arguments.push_back(arg);
    }
    parsed.kernels.push_back(std::move(kernel));
  }
  uint32_t sourceSize;
  if (!u32(sourceSize) || !sourceSize || !text(parsed.source, sourceSize) ||
      cursor != size)
    return false;
  output = std::move(parsed);
  return true;
}
inline size_t nativeArgumentCount(const KernelABI &kernel) {
  size_t count = (kernel.dynamicShared ? 1 : 0) + (kernel.deviceErrors ? 1 : 0);
  for (const auto &arg : kernel.arguments)
    for (const auto &field : arg.fields)
      count += field.kind == ArgumentABI::BufferView ? 2 : 1;
  return count;
}
} // namespace M::OpenCL
#endif
