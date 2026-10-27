// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#ifndef MOJO_OPENCL_EMITTER_H
#define MOJO_OPENCL_EMITTER_H
#include "Target/OpenCLBundle.h"
#include "llvm/Support/Error.h"
namespace llvm {
class Module;
}
namespace M::KGEN {
// Lower a checked LLVM kernel subset to OpenCL C and its versioned ABI.
// Unsupported operations fail compilation; no fallback code is emitted.
llvm::Expected<M::OpenCL::Bundle> emitOpenCL(llvm::Module &module);
} // namespace M::KGEN
#endif
