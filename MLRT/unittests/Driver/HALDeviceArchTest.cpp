// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#include "MLRT/Driver/DeviceContext/HALDeviceArch.h"
#include "gtest/gtest.h"
#include <cstring>
#include <utility>

namespace {
struct Plugin {
  const char *api;
  const char *arch;
  int attributeCalls = 0;
  int major = 9;
  int minor = 0;
  bool badArch = false;

  static M_driver_status_t property(M_driver_driver_t d, const char *,
                                    void *v) {
    auto &p = *reinterpret_cast<Plugin *>(d);
    *static_cast<const char **>(v) = p.api;
    return M_DRIVER_STATUS_SUCCESS;
  }
  static M_driver_status_t deviceProperty(M_driver_device_t d, const char *,
                                          void *v) {
    auto &p = *reinterpret_cast<Plugin *>(d);
    if (!p.arch)
      return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
    if (p.badArch)
      std::memset(v, 'x', M_DRIVER_DEVICE_PROPERTY_MAX_LEN);
    else
      std::strcpy(static_cast<char *>(v), p.arch);
    return M_DRIVER_STATUS_SUCCESS;
  }
  static M_driver_status_t attribute(M_driver_device_t d, int32_t a,
                                     int32_t *v) {
    auto &p = *reinterpret_cast<Plugin *>(d);
    ++p.attributeCalls;
    *v = a == 75 ? p.major : p.minor;
    return M_DRIVER_STATUS_SUCCESS;
  }
  std::string query() {
    return M::Driver::detail::queryHALDeviceArch(
        reinterpret_cast<M_driver_driver_t>(this),
        reinterpret_cast<M_driver_device_t>(this), property, deviceProperty,
        attribute);
  }
};

TEST(HALDeviceArchTest, ExplicitPluginArchitectures) {
  for (auto pair :
       {std::pair{"OpenCL", "adreno-840"},
        std::pair{"OpenCL", "opencl-generic"}, std::pair{"HIP", "gfx942"},
        std::pair{"CUDA", "sm_100f"}, std::pair{"Metal", "apple-m1"}}) {
    Plugin p{pair.first, pair.second};
    std::string expected = std::string(pair.first) == "HIP"     ? "amdgpu:"
                           : std::string(pair.first) == "CUDA"  ? "nvidia:"
                           : std::string(pair.first) == "Metal" ? "metal:"
                                                                : "opencl:";
    EXPECT_EQ(p.query(), expected + pair.second);
    EXPECT_EQ(p.attributeCalls, 0);
  }
}

TEST(HALDeviceArchTest, MissingNonCudaArchDoesNotUseCudaAttributes) {
  Plugin p{"OpenCL", nullptr};
  EXPECT_EQ(p.query(), "");
  EXPECT_EQ(p.attributeCalls, 0);
  p.arch = "";
  EXPECT_EQ(p.query(), "");
  p.badArch = true;
  EXPECT_EQ(p.query(), "");
  EXPECT_EQ(p.attributeCalls, 0);
}

TEST(HALDeviceArchTest, LegacyCudaFallback) {
  Plugin p{"CUDA", nullptr};
  EXPECT_EQ(p.query(), "nvidia:sm_90a");
  p.major = 8;
  p.minor = 6;
  EXPECT_EQ(p.query(), "nvidia:sm_86");
  p.major = 0;
  EXPECT_EQ(p.query(), "");
}

TEST(HALDeviceArchTest, OptionalSymbols) {
  Plugin p{"OpenCL", "adreno-840"};
  auto driver = reinterpret_cast<M_driver_driver_t>(&p);
  auto device = reinterpret_cast<M_driver_device_t>(&p);
  EXPECT_EQ(
      M::Driver::detail::queryHALDeviceArch(driver, device, Plugin::property,
                                            Plugin::deviceProperty, nullptr),
      "opencl:adreno-840");
  EXPECT_EQ(M::Driver::detail::queryHALDeviceArch(driver, device, nullptr,
                                                  nullptr, Plugin::attribute),
            "nvidia:sm_90a");
  EXPECT_EQ(M::Driver::detail::queryHALDeviceArch(
                driver, device, Plugin::property, nullptr, nullptr),
            "");
}
} // namespace
