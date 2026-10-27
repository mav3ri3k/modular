// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#include "Target/OpenCL/OpenCLTraits.h"
namespace M::KGEN {
const OpenCLTraits &OpenCLTraits::get() {
  static const OpenCLTraits traits;
  return traits;
}
namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<OpenCLTraits> registerOpenCL;
#pragma GCC diagnostic pop
} // namespace
} // namespace M::KGEN
