// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"
#include "gtest/gtest.h"
using namespace M;
using namespace M::AsyncRT;
using namespace M::Driver;
using M::MLRT::Tensor;
using M::MLRT::TensorBufferRef;
extern "C" AsyncValue *test_opaque_graph_tensor(AsyncValue *, void *);
extern "C" int32_t test_opaque_graph_tensor_read(AsyncValue *, void *, float *);

TEST(OpaqueGraphTensor, ShapeSlicePackAndUnpackPreserveOpaqueOwner) {
  ASSIGN_NO_ERR(DeviceContextRef, cpu, DeviceContext::create(Device::cpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, gpu, DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceBufferRef, allocation, gpu->enqueueCreateBuffer(64));
  ASSIGN_NO_ERR(DeviceBuffer::OpaqueMemoryView, view,
                allocation->getOpaqueMemoryView());
  auto &runtime = *CPUDevice::getCurrentCPUDeviceOrNull();
  auto storage = AnyAsyncValueRef::createReady<DeviceBufferRef>(
      runtime, std::move(allocation));
  auto buffer = TensorBufferRef::createOpaque({view.allocation, 0, 64},
                                              std::move(storage));
  auto packed = AnyAsyncValueRef::createReady<Tensor>(
      runtime, std::move(buffer), TensorSpec(TensorShape({2, 8}), DType::f32));
  auto result = AnyAsyncValueRef::take(
      test_opaque_graph_tensor(packed.releasePointer(), gpu->mojoHandle()));
  ASSERT_TRUE(result);
  auto &tensor = result.get<Tensor>();
  EXPECT_EQ(tensor.getSpec()[0], 2u);
  EXPECT_EQ(tensor.getSpec()[1], 4u);
  const auto &nested = tensor.getBufferRef().getOpaqueView();
  EXPECT_EQ(nested.allocation, view.allocation);
  EXPECT_EQ(nested.byteOffset, 16u);
  EXPECT_EQ(nested.byteSize, 32u);
  float actual[8]{};
  EXPECT_EQ(test_opaque_graph_tensor_read(result.releasePointer(),
                                          gpu->mojoHandle(), actual),
            0);
  for (float value : actual)
    EXPECT_EQ(value, 2.0f);
}
