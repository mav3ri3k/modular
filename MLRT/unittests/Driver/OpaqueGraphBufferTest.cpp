// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::AsyncRT;
using namespace M::Driver;
using M::MLRT::TensorBufferRef;
extern "C" AsyncValue *test_opaque_graph_slice(AsyncValue *, void *);
extern "C" int32_t test_opaque_graph_fill_read(AsyncValue *, void *, float *);

TEST(OpaqueGraphBuffer, NestedSlicePackUnpackAndFillPreserveStorage) {
  ASSIGN_NO_ERR(DeviceContextRef, cpu, DeviceContext::create(Device::cpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, gpu, DeviceContext::create(Device::gpuAPI));
  ASSERT_EQ(gpu->device().api(), Device::openclAPI);
  EXPECT_EQ(gpu->device().getAcceleratorArchOrEmpty(), "opencl:mock");
  ASSIGN_NO_ERR(DeviceBufferRef, allocation, gpu->enqueueCreateBuffer(64));
  ASSERT_TRUE(allocation->hasOpaqueStorage());
  ASSIGN_NO_ERR(DeviceBuffer::OpaqueMemoryView, view,
                allocation->getOpaqueMemoryView());
  auto &runtime = *CPUDevice::getCurrentCPUDeviceOrNull();
  auto storage = AnyAsyncValueRef::createReady<DeviceBufferRef>(
      runtime, std::move(allocation));
  auto buffer = TensorBufferRef::createOpaque(
      {view.allocation, view.byteOffset, view.byteSize}, std::move(storage));
  auto packed = AnyAsyncValueRef::createReady<TensorBufferRef>(
      runtime, std::move(buffer));
  auto nested = AnyAsyncValueRef::take(
      test_opaque_graph_slice(packed.releasePointer(), gpu->mojoHandle()));
  ASSERT_TRUE(nested);
  auto &result = nested.get<TensorBufferRef>();
  ASSERT_TRUE(result.isOpaque());
  EXPECT_EQ(result.getOpaqueView().allocation, view.allocation);
  EXPECT_EQ(result.getOpaqueView().byteOffset, 12u);
  EXPECT_EQ(result.getOpaqueView().byteSize, 16u);
  // The original buffer/packed value is gone. Only the returned view owns it.
  float actual[4] = {};
  EXPECT_EQ(test_opaque_graph_fill_read(nested.releasePointer(),
                                        gpu->mojoHandle(), actual),
            0);
  for (float value : actual)
    EXPECT_EQ(value, 2.0f);
}
