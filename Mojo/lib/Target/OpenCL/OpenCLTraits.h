// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#ifndef KGEN_TARGET_OPENCL_TRAITS_H
#define KGEN_TARGET_OPENCL_TRAITS_H
#include "Target/TargetTraits.h"
#include "llvm/TargetParser/Triple.h"
namespace M::KGEN {
// AArch64 supplies the initial 64-bit little-endian scalar data model only.
// No ARM machine instructions are emitted for this device target.
struct OpenCLTraits final : TargetTraits {
  llvm::StringRef name() const override { return "opencl"; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.isAArch64() && triple.getOSName() == "opencl";
  }
  bool isGPU() const override { return true; }
  llvm::StringRef defaultCPU(const llvm::Triple &) const override {
    return "generic";
  }
  llvm::StringRef getAsmExtension() const override { return ".cl"; }
  llvm::StringRef getLLVMExtension() const override { return ".opencl.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".mcl"; }
  llvm::StringRef getBitcodeExtension() const override { return ".bc"; }
  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    static const EmissionKind kinds[] = {
        {"object", EmitAs::OBJECT, "a versioned OpenCL kernel bundle"},
        {"asm", EmitAs::ASM, "generated OpenCL C"},
        {"llvm", EmitAs::LLVM, "OpenCL-target LLVM IR"},
        {"llvm-bitcode", EmitAs::LLVM_BITCODE, "OpenCL-target LLVM bitcode"},
    };
    return kinds;
  }
  bool isBaseTarget() const override { return true; }
  llvm::StringRef acceleratorSectionTitle() const override {
    return "OpenCL GPU:";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override {
    static const AcceleratorArch archs[] = {
        {"opencl-generic", "OpenCL 1.2 baseline"},
        {"adreno-840", "Adreno 840 with validated 64-lane subgroups"},
    };
    return archs;
  }
  static const OpenCLTraits &get();
};
} // namespace M::KGEN
#endif
