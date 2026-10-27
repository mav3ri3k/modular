// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#ifndef MLRT_DRIVER_DEVICECONTEXT_HALDEVICEARCH_H
#define MLRT_DRIVER_DEVICECONTEXT_HALDEVICEARCH_H

#include "HAL/M_driver_api.h"
#include "MLRT/Driver/DeviceContext/DeviceAPINames.h"
#include <cstring>
#include <string>

namespace M::Driver::detail {

/// Query a compile target without linking a device-context runtime. The
/// callbacks also allow callers that load optional HAL symbols dynamically.
inline std::string
queryHALDeviceArch(M_driver_driver_t driver, M_driver_device_t device,
                   M_driver_property_func_ptr_t property,
                   M_driver_device_property_func_ptr_t deviceProperty,
                   M_driver_device_attribute_func_ptr_t deviceAttribute) {
  const char *apiName = nullptr;
  std::string api;
  if (property &&
      property(driver, M_DRIVER_PROPERTY_API, &apiName) ==
          M_DRIVER_STATUS_SUCCESS &&
      apiName)
    api = canonicalDeviceAPI(llvm::StringRef(apiName).lower()).str();

  // Prefer the plugin's explicit architecture. API names are case-insensitive
  // in the Mojo HAL; architecture names retain the plugin's exact spelling.
  char arch[M_DRIVER_DEVICE_PROPERTY_MAX_LEN] = {};
  if (!api.empty() && deviceProperty &&
      deviceProperty(device, M_DRIVER_DEVICE_PROPERTY_ARCH, arch) ==
          M_DRIVER_STATUS_SUCCESS &&
      arch[0] && std::memchr(arch, '\0', sizeof(arch)))
    return apiToVendorPrefix(api).str() + ":" + arch;

  // Compatibility with CUDA plugins predating the architecture property (or
  // the API property). Never ask a known non-CUDA backend for CUDA attributes.
  if ((!api.empty() && api != "cuda") || !deviceAttribute)
    return {};
  int32_t major = 0, minor = 0;
  if (deviceAttribute(device, 75, &major) != M_DRIVER_STATUS_SUCCESS ||
      deviceAttribute(device, 76, &minor) != M_DRIVER_STATUS_SUCCESS ||
      major <= 0 || minor < 0 || minor > 9 || major > 99)
    return {};
  int cc = major * 10 + minor;
  return "nvidia:sm_" + std::to_string(cc) + (cc >= 90 ? "a" : "");
}

} // namespace M::Driver::detail
#endif
