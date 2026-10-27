// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::Driver;

namespace {
class OpenCLVirtualDeviceTest : public testing::Test {
  void SetUp() override {
    AsyncRT::setVirtualDeviceAPI("opencl");
    AsyncRT::setVirtualDeviceTargetArch("adreno-840");
    AsyncRT::setVirtualDeviceCount(1);
  }
  void TearDown() override {
    AsyncRT::setVirtualDeviceCount(0);
    AsyncRT::setVirtualDeviceAPI("");
    AsyncRT::setVirtualDeviceTargetArch("");
  }
};

TEST_F(OpenCLVirtualDeviceTest, ConcreteAndGenericApiWithoutPhysicalDriver) {
  for (auto api : {Device::gpuAPI, Device::openclAPI}) {
    ASSIGN_NO_ERR(DeviceRef, device, Device::create(api, 0));
    EXPECT_EQ(device->api(), Device::openclAPI);
    ASSIGN_NO_ERR(const char *, arch, device->getArchName());
    EXPECT_STREQ(arch, "adreno-840");
    ASSIGN_NO_ERR(DeviceContextRef, context, device->createContext());
    EXPECT_NO_ERR(context->synchronize());
#if MODULAR_HAL_DEVICE_CONTEXT
    EXPECT_EQ(device->getDLDevice(false).device_type, kDLOpenCL);
    EXPECT_EQ(Device::preferredAlignment(Device::openclAPI), 256u);
#endif
  }
}
} // namespace
