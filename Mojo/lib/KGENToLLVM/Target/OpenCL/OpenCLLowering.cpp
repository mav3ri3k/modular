// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#include "KGENToLLVM/LLVMLoweringUtils.h"
#include "Target/OpenCL/OpenCLTraits.h"
#include "Target/TargetLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
namespace M::KGEN {
namespace {
class OpenCLLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &OpenCLTraits::get(); }
  bool isBaseTarget() const override { return true; }
  bool usesHostCABI() const override { return false; }
  llvm::StringRef getKernelByValArgAttrName() const override {
    return "llvm.byval";
  }
  void markExportedKernel(mlir::Operation *op) const override {
    mlir::cast<mlir::LLVM::LLVMFuncOp>(op).setCConv(
        mlir::LLVM::CConv::SPIR_KERNEL);
  }
  mlir::LogicalResult
  mapFuncMetadata(mlir::Operation *func, mlir::NamedAttribute attr,
                  mlir::NamedAttrList &,
                  llvm::SmallVectorImpl<mlir::Attribute> &passthrough,
                  bool &handled) const override {
    handled = attr.getName() == "pop.opencl.max_work_group_size";
    if (!handled)
      return mlir::success();
    auto range = getWorkGroupSizeRange(attr.getValue());
    if (range.isError())
      return mlir::emitError(func->getLoc()) << range.takeError();
    mlir::Builder builder(func->getContext());
    passthrough.push_back(builder.getArrayAttr(
        {builder.getStringAttr("mojo.opencl.work_group_size_hint"),
         builder.getStringAttr(std::to_string(range.get().second))}));
    return mlir::success();
  }
  bool isExportedKernel(mlir::Operation *op) const override {
    auto fn = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op);
    return fn && fn.getCConv() == mlir::LLVM::CConv::SPIR_KERNEL;
  }
};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<OpenCLLowering> registerOpenCL;
#pragma GCC diagnostic pop
} // namespace
} // namespace M::KGEN
