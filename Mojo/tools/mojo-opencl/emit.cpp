// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#include "OpenCLEmitter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
int main(int argc, char **argv) {
  bool markExported =
      argc > 1 && llvm::StringRef(argv[argc - 1]) == "--mark-exported";
  if (markExported)
    --argc;
  if (argc != 3 && argc != 4) {
    llvm::errs() << "usage: mojo-opencl-emit input.ll output.mcl "
                    "[generated.cl] [--mark-exported]\n";
    return 2;
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIRFile(argv[1], diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }
  // Bootstrap with a released compiler that does not yet register OpenCL's
  // kernel calling convention. This flag is explicit and restricted to the
  // OpenCL triple; it never treats an ordinary host module as device code.
  if (markExported) {
    if (module->getTargetTriple().getOSName() != "opencl") {
      llvm::errs() << "--mark-exported requires an OpenCL-target module\n";
      return 1;
    }
    for (auto &fn : *module)
      if (!fn.isDeclaration() && fn.hasExternalLinkage())
        fn.addFnAttr("mojo.opencl.kernel");
  }
  auto bundle = M::KGEN::emitOpenCL(*module);
  if (!bundle) {
    llvm::errs() << llvm::toString(bundle.takeError()) << "\n";
    return 1;
  }
  std::error_code error;
  llvm::ToolOutputFile output(argv[2], error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << error.message() << "\n";
    return 1;
  }
  output.os() << M::OpenCL::encodeBundle(*bundle);
  if (argc == 4) {
    llvm::ToolOutputFile source(argv[3], error, llvm::sys::fs::OF_Text);
    if (error) {
      llvm::errs() << error.message() << "\n";
      return 1;
    }
    source.os() << bundle->source;
    source.keep();
  }
  output.keep();
  llvm::outs() << "Compiled " << bundle->kernels.size()
               << " Mojo kernel(s) for OpenCL ABI v" << bundle->version << "\n";
}
