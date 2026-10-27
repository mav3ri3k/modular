// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "OpenCLEmitter.h"
#include "Target/OpenCL/OpenCLTraits.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
namespace M::KGEN {
namespace {
class OpenCLBackend final : public TargetBackend {
public:
  const TargetTraits *traits() const override { return &OpenCLTraits::get(); }
  bool isBaseTarget() const override { return true; }
  bool isOffload() const override { return true; }
  bool usesModuleEmitterForArchive() const override { return true; }
  SplitStrategy splitStrategy(const CompilationOptions &) const override {
    return SplitStrategy::None;
  }
  bool isCodegenInterprocedural() const override { return true; }
  // Run the shared optimization pipeline, without the borrowed AArch64 ISA
  // cost model. Vendor OpenCL compilation performs final device instruction
  // selection; generic LLVM simplification/inlining/loop passes still apply.
  bool usesTargetMachineForOptimization() const override { return false; }
  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef) const override {
    auto result = options;
    result.targetTriple = "aarch64-unknown-linux-gnu";
    result.targetCpu = "generic";
    result.targetFeatures = "";
    return result;
  }
  void finalizeModuleForTarget(llvm::Module &module, llvm::TargetMachine &,
                               llvm::StringRef originalTriple) const override {
    module.setTargetTriple(llvm::Triple(originalTriple));
    // OpenCL collectives are ordinary external calls in portable LLVM IR.
    // Their convergence semantics must be visible before the shared optimizer
    // runs, just as for LLVM's native GPU collective intrinsics.
    for (auto &function : module) {
      auto name = function.getName();
      if (name != "__mojo_opencl_shuffle_idx_u32_w64" &&
          name != "__mojo_opencl_barrier" &&
          name != "__mojo_opencl_subgroup_barrier_w64" &&
          name != "__mojo_opencl_subgroup_local_id")
        continue;
      function.addFnAttr(llvm::Attribute::Convergent);
      function.addFnAttr(llvm::Attribute::NoDuplicate);
      for (auto *user : function.users())
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(user)) {
          call->addFnAttr(llvm::Attribute::Convergent);
          call->addFnAttr(llvm::Attribute::NoDuplicate);
        }
    }
  }
  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &ctx) const override {
    if (ctx.options.sanitizers)
      return Error("OpenCL: device sanitizers are not implemented");
    auto output = emitOpenCL(module);
    if (!output)
      return Error(llvm::toString(output.takeError()));
    auto buffer = WriteableBuffer::get();
    *buffer << output->source;
    return buffer;
  }
  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override {
    if (ctx.options.sanitizers)
      return Error("OpenCL: device sanitizers are not implemented");
    auto output = emitOpenCL(module);
    if (!output)
      return Error(llvm::toString(output.takeError()));
    auto buffer = WriteableBuffer::get();
    *buffer << M::OpenCL::encodeBundle(*output);
    return buffer;
  }
  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef,
                                   EmitContext &) const override {
    if (objects.size() != 1)
      return Error("OpenCL: multiple compilation units are unsupported");
    return objects.front().copy();
  }
};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<OpenCLBackend> registerOpenCL;
#pragma GCC diagnostic pop
} // namespace
} // namespace M::KGEN
