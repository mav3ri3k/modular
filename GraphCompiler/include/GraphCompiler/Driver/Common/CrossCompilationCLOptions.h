// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#ifndef GRAPHCOMPILER_DRIVER_COMMON_CROSSCOMPILATIONCLOPTIONS_H
#define GRAPHCOMPILER_DRIVER_COMMON_CROSSCOMPILATIONCLOPTIONS_H

#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace M {

/// Exposes the driver's existing compile-only device settings to native tools.
class CrossCompilationCLOptions {
  llvm::cl::OptionCategory category{"Cross-compilation options"};
  llvm::cl::opt<std::string> cpu{
      "virtual-cpu-target",
      llvm::cl::desc("CPU codegen override: CPU name or "
                     "triple=...;cpu=...[;features=...]"),
      llvm::cl::cat(category)};
  llvm::cl::opt<int> count{"virtual-device-count",
                           llvm::cl::desc("Compile-only accelerator count"),
                           llvm::cl::init(0), llvm::cl::cat(category)};
  llvm::cl::opt<std::string> api{"virtual-device-api",
                                 llvm::cl::desc("Compile-only accelerator API"),
                                 llvm::cl::cat(category)};
  llvm::cl::opt<std::string> arch{
      "virtual-device-arch",
      llvm::cl::desc("Compile-only accelerator architecture"),
      llvm::cl::cat(category)};

public:
  /// Applies validated settings before a compilation creates devices.
  bool apply() const {
    if (count < 0 ||
        (count > 0 && (api.getValue().empty() || arch.getValue().empty())) ||
        (count == 0 && (!api.getValue().empty() || !arch.getValue().empty()))) {
      llvm::errs() << "error: virtual-device-count must be positive when "
                      "virtual-device-api and virtual-device-arch are set\n";
      return false;
    }
    if (!cpu.getValue().empty())
      AsyncRT::setVirtualCpuTarget(cpu.getValue());
    if (count > 0) {
      AsyncRT::setVirtualDeviceAPI(api.getValue());
      AsyncRT::setVirtualDeviceTargetArch(arch.getValue());
      AsyncRT::setVirtualDeviceCount(count);
    }
    return true;
  }
};

} // namespace M
#endif
