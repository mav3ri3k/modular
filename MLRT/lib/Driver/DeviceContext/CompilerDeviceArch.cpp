//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Minimal, plugin-direct implementation of
// `M::Driver::Device::getAcceleratorArchOrEmpty()` for the Mojo/KGEN compiler
// family.
//
// The compiler only needs the local accelerator's arch string (to set the
// compilation target); it must NOT link a device-context runtime. The HAL
// DeviceContext implementation is Mojo-backed (it reaches `@mojo//:std`), which
// would form a build-time bootstrap cycle; the legacy C++ DeviceContext runtime
// is not linked either. This TU provides just that one static by `dlopen`ing
// the driver plugin named in `MODULAR_DRIVER_PLUGINS` and querying device 0's
// API and architecture over the `M_driver_*` C ABI directly. It returns the
// `vendor:arch` form expected by the target registry.
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "HAL/M_driver_api.h"
#include "MLRT/Driver/DeviceContext/DeviceAPINames.h"
#include "MLRT/Driver/DeviceContext/HALDeviceArch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"

#include <cstdint>
#include <dlfcn.h>
#include <optional>
#include <string>

namespace M {
namespace Driver {

namespace {

// `dlopen`s the plugin from `MODULAR_DRIVER_PLUGINS`
// ("Name@/path/to/plugin.so") and queries device 0's API and architecture.
// Returns "" when
// no plugin/device is available (e.g. a GPU-less build host); callers then fall
// back to an explicit target arch, exactly as with the previous legacy
// DeviceContext implementation.
std::string queryAcceleratorArch() {
  std::optional<std::string> spec =
      llvm::sys::Process::GetEnv("MODULAR_DRIVER_PLUGINS");
  if (!spec || spec->empty())
    return {};

  // The spec is "Name@/path/to/plugin.so"; the path follows the '@' (or is the
  // whole string when there is no '@').
  llvm::StringRef soPath(*spec);
  if (size_t at = soPath.find('@'); at != llvm::StringRef::npos)
    soPath = soPath.substr(at + 1);

  void *lib = ::dlopen(std::string(soPath).c_str(), RTLD_NOW);
  if (!lib)
    return {};

  auto create = reinterpret_cast<M_driver_create_func_ptr_t>(
      ::dlsym(lib, "M_driver_create"));
  auto destroy = reinterpret_cast<M_driver_destroy_func_ptr_t>(
      ::dlsym(lib, "M_driver_destroy"));
  auto deviceGet = reinterpret_cast<M_driver_device_get_func_ptr_t>(
      ::dlsym(lib, "M_driver_device_get"));
  auto deviceAttribute = reinterpret_cast<M_driver_device_attribute_func_ptr_t>(
      ::dlsym(lib, "M_driver_device_attribute"));
  auto property = reinterpret_cast<M_driver_property_func_ptr_t>(
      ::dlsym(lib, "M_driver_property"));
  auto deviceProperty = reinterpret_cast<M_driver_device_property_func_ptr_t>(
      ::dlsym(lib, "M_driver_device_property"));
  if (!create || !destroy || !deviceGet) {
    ::dlclose(lib);
    return {};
  }

  M_driver_version version{M_DRIVER_INTERFACE_VERSION_MAJOR,
                           M_DRIVER_INTERFACE_VERSION_MINOR,
                           M_DRIVER_INTERFACE_VERSION_PATCH};
  M_driver_driver_t driver = nullptr;
  if (create(&version, &driver) != M_DRIVER_STATUS_SUCCESS || !driver) {
    ::dlclose(lib);
    return {};
  }

  std::string arch;
  M_driver_device_t device = nullptr;
  if (deviceGet(driver, /*device_id=*/0, &device) == M_DRIVER_STATUS_SUCCESS &&
      device)
    arch = detail::queryHALDeviceArch(driver, device, property, deviceProperty,
                                      deviceAttribute);

  destroy(driver);
  ::dlclose(lib);
  return arch;
}

} // namespace

const std::string &Device::getAcceleratorArchOrEmpty() {
  static const std::string arch = [] {
    // Virtual (compile-only) mode wins over plugin detection: with the knobs
    // set, the target arch is the configured one and there is deliberately no
    // accelerator to query. Without this the compiler would fall back to "" on
    // a GPU-less host and silently compile for the wrong (or no) target.
    if (std::string virtualArch = AsyncRT::getVirtualDeviceTargetArch();
        !virtualArch.empty())
      return (llvm::Twine(apiToVendorPrefix(AsyncRT::getVirtualDeviceAPI())) +
              ":" + virtualArch)
          .str();
    return queryAcceleratorArch();
  }();
  return arch;
}

// `DeviceContext::formatError` is the error formatter behind `ErrorMacros.h`,
// reached by `Driver/CompilationDeviceImpl`. We provide an implementation here
// for targets that don't like a DeviceContext implementation at all.
Error DeviceContext::formatError(const char *file, int line, const char *func,
                                 const llvm::Twine &msg) {
  return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
               " failed: " + msg);
}

} // namespace Driver
} // namespace M
