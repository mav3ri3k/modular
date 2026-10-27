// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::Driver;

TEST(OpenCLCompilationDeviceTest, NoPhysicalDriverRequired) {
  for (const char *arch : {"opencl-generic", "adreno-840"}) {
    ASSIGN_NO_ERR(std::unique_ptr<CompilationDevice>, device,
                  CompilationDevice::create("opencl", arch));
    EXPECT_EQ(device->api(), "opencl");
    EXPECT_EQ(device->targetArch(), arch);
    ASSIGN_NO_ERR(const char *, storedArch, device->getArchName());
    EXPECT_STREQ(storedArch, arch);
    EXPECT_ERR_MSG(device->getApiVersion(), "unavailable in compile-only mode");
    EXPECT_ERR_MSG(device->getDriverVersion(),
                   "unavailable in compile-only mode");
    EXPECT_ERR_MSG(device->compileFunction("", "", -1),
                   "KGEN emits the bundle");
  }
}

TEST(OpenCLCompilationDeviceTest, RequiresArchitecture) {
  EXPECT_ERR_MSG(CompilationDevice::create("opencl", ""),
                 "require a target architecture");
}
