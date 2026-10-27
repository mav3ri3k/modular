// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Core/Tensor/StateSlot.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"
#include "gtest/gtest.h"
#include <array>

using namespace M;
using namespace M::AsyncRT;
using M::MLRT::TensorBufferRef;
namespace {
struct AllocationOwner {
  explicit AllocationOwner(int *drops) : drops(drops) {}
  ~AllocationOwner() { ++*drops; }
  int *drops;
};

TEST(OpaqueTensorBufferRef,
     StateSlotCachesOpaqueViewsWithoutRequestingAddresses) {
  ASSIGN_NO_ERR(Driver::DeviceContextRef, context,
                Driver::DeviceContext::create(Driver::Device::cpuAPI));
  auto &cpu = *CPUDevice::getCurrentCPUDeviceOrNull();
  int drops = 0;
  auto slot = RCRef<MLRT::TensorStateSlot>::create();
  EXPECT_ERR(slot->getBuffer());
  EXPECT_ERR(slot->getBufferRef());
  auto owner = AnyAsyncValueRef::createReady<AllocationOwner>(cpu, &drops);
  EXPECT_NO_ERR(slot->setBuffer(
      TensorBufferRef::createOpaque({7, 12, 16}, owner.copy())));
  owner.reset();
  EXPECT_EQ(drops, 0);
  auto reference = slot->getBufferRef();
  ASSERT_FALSE(reference.isError());
  EXPECT_EQ(reference.get().getOpaqueView().byteOffset, 12u);
  auto borrowed = slot->getBuffer();
  ASSERT_FALSE(borrowed.isError());
  EXPECT_EQ(borrowed.get().getOpaqueView().allocation, 7u);
  EXPECT_EQ(borrowed.get().getSize(), 16u);
  EXPECT_ERR(slot->setBuffer({}));
}

TEST(OpaqueTensorBufferRef, BorrowSwapAndMoveRetainAllocationIdentityAndOwner) {
  ASSIGN_NO_ERR(Driver::DeviceContextRef, context,
                Driver::DeviceContext::create(Driver::Device::cpuAPI));
  auto &cpu = *CPUDevice::getCurrentCPUDeviceOrNull();
  int drops = 0;
  auto owner = AnyAsyncValueRef::createReady<AllocationOwner>(cpu, &drops);
  auto buffer = TensorBufferRef::createOpaque({7, 12, 16}, owner.copy());
  auto borrowed = buffer.borrow();
  buffer = {};
  owner.reset();
  EXPECT_EQ(drops, 0);
  ASSERT_TRUE(borrowed.isOpaque());
  EXPECT_EQ(borrowed.getOpaqueView().allocation, 7u);
  EXPECT_EQ(borrowed.getOpaqueView().byteOffset, 12u);
  EXPECT_EQ(borrowed.getSize(), 16u);

  std::array<char, 4> bytes{};
  auto raw =
      TensorBufferRef::createWithNonTrackedMemory(bytes.data(), bytes.size());
  raw.swap(borrowed);
  EXPECT_TRUE(raw.isOpaque());
  EXPECT_FALSE(borrowed.isOpaque());
  EXPECT_EQ(borrowed.getBuffer(), bytes.data());
  EXPECT_EQ(borrowed.getSize(), bytes.size());
  auto moved = std::move(raw);
  EXPECT_FALSE(raw.isOpaque());
  EXPECT_EQ(moved.getOpaqueView().byteOffset, 12u);
  EXPECT_EQ(drops, 0);
  moved = {};
  EXPECT_EQ(drops, 1);
}
} // namespace
