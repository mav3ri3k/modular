// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// Experimental OpenCL implementation of the vendor-facing MAX driver HAL.
#define CL_TARGET_OPENCL_VERSION 120
#include "HAL/M_driver_api.h"
#include "Target/OpenCLBundle.h"
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#if defined(__ANDROID__) || defined(__APPLE__) || defined(__linux__)
#include <execinfo.h>
#endif
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#define CL_FUNCTIONS(X)                                                        \
  X(clGetPlatformIDs)                                                          \
  X(clGetPlatformInfo)                                                        \
  X(clGetExtensionFunctionAddressForPlatform)                                 \
  X(clGetDeviceIDs)                                                            \
  X(clGetDeviceInfo)                                                           \
  X(clCreateContext)                                                           \
  X(clReleaseContext)                                                          \
  X(clCreateCommandQueue)                                                      \
  X(clReleaseCommandQueue)                                                     \
  X(clEnqueueMapBuffer)                                                        \
  X(clEnqueueUnmapMemObject)                                                   \
  X(clCreateBuffer)                                                            \
  X(clRetainMemObject)                                                         \
  X(clReleaseMemObject)                                                        \
  X(clEnqueueWriteBuffer)                                                      \
  X(clEnqueueReadBuffer)                                                       \
  X(clEnqueueCopyBuffer)                                                       \
  X(clEnqueueFillBuffer)                                                       \
  X(clCreateProgramWithSource)                                                 \
  X(clBuildProgram)                                                            \
  X(clGetProgramBuildInfo)                                                     \
  X(clReleaseProgram)                                                          \
  X(clCreateKernel)                                                            \
  X(clReleaseKernel)                                                           \
  X(clGetKernelInfo)                                                           \
  X(clGetKernelWorkGroupInfo)                                                  \
  X(clSetKernelArg)                                                            \
  X(clEnqueueNDRangeKernel)                                                    \
  X(clFinish)                                                                  \
  X(clFlush)                                                                   \
  X(clEnqueueMarkerWithWaitList)                                               \
  X(clEnqueueBarrierWithWaitList)                                              \
  X(clRetainEvent)                                                             \
  X(clSetEventCallback)                                                        \
  X(clReleaseEvent)                                                            \
  X(clWaitForEvents)                                                           \
  X(clGetEventInfo)                                                            \
  X(clGetEventProfilingInfo)

struct OpenCL {
  void *library = nullptr;
  using Recording = struct _cl_recording_qcom *;
  struct ArrayArg {
    cl_uint dispatch_index;
    cl_uint arg_index;
    size_t arg_size;
    const void *arg_value;
  };
  struct Workgroup {
    cl_uint dispatch_index;
    const size_t *workgroup_size;
  };
  using NewRecording = Recording(CL_API_CALL *)(cl_command_queue, cl_int *);
  using EndRecording = cl_int(CL_API_CALL *)(Recording);
  using ReleaseRecording = cl_int(CL_API_CALL *)(Recording);
  using EnqueueRecording = cl_int(CL_API_CALL *)(
      cl_command_queue, Recording, size_t, const ArrayArg *, size_t,
      const void *, size_t, const Workgroup *, size_t, const Workgroup *,
      cl_uint, const cl_event *, cl_event *);
  NewRecording newRecording = nullptr;
  EndRecording endRecording = nullptr;
  ReleaseRecording releaseRecording = nullptr;
  EnqueueRecording enqueueRecording = nullptr;
  using SubgroupInfo = cl_int(CL_API_CALL *)(cl_kernel, cl_device_id, cl_uint,
                                             size_t, const void *, size_t,
                                             void *, size_t *);
  SubgroupInfo subgroupInfo = nullptr;
#define DECLARE(name) decltype(&::name) name = nullptr;
  CL_FUNCTIONS(DECLARE)
#undef DECLARE
  bool load() {
    library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library)
      library = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library)
      return false;
#define LOAD(name)                                                             \
  name = reinterpret_cast<decltype(name)>(dlsym(library, #name));              \
  if (!name)                                                                   \
    return false;
    CL_FUNCTIONS(LOAD)
#undef LOAD
    subgroupInfo = reinterpret_cast<SubgroupInfo>(
        dlsym(library, "clGetKernelSubGroupInfo"));
    if (!subgroupInfo)
      subgroupInfo = reinterpret_cast<SubgroupInfo>(
          dlsym(library, "clGetKernelSubGroupInfoKHR"));
    cl_uint platformCount = 0;
    if (!clGetPlatformIDs(0, nullptr, &platformCount) && platformCount) {
      std::vector<cl_platform_id> platforms(platformCount);
      if (!clGetPlatformIDs(platformCount, platforms.data(), nullptr)) {
        newRecording = reinterpret_cast<NewRecording>(
            clGetExtensionFunctionAddressForPlatform(
                platforms[0], "clNewRecordingQCOM"));
        endRecording = reinterpret_cast<EndRecording>(
            clGetExtensionFunctionAddressForPlatform(
                platforms[0], "clEndRecordingQCOM"));
        releaseRecording = reinterpret_cast<ReleaseRecording>(
            clGetExtensionFunctionAddressForPlatform(
                platforms[0], "clReleaseRecordingQCOM"));
        enqueueRecording = reinterpret_cast<EnqueueRecording>(
            clGetExtensionFunctionAddressForPlatform(
                platforms[0], "clEnqueueRecordingQCOM"));
      }
    }
    return true;
  }
  ~OpenCL() {
    if (library)
      dlclose(library);
  }
};
struct M_driver_device {
  OpenCL *api;
  cl_device_id native;
  int32_t ordinal;
};
struct M_driver_driver {
  OpenCL api;
  std::vector<M_driver_device> devices;
};
struct M_driver_context {
  M_driver_device_t device;
  cl_context native;
  cl_command_queue transfer;
  std::mutex memoryMutex{};
  std::set<M_driver_memory_t> memories{};
};
struct M_driver_memory {
  M_driver_context_t context;
  cl_mem native;
  uint64_t size;
  void *hostAddress = nullptr;
};
struct M_driver_captured_arg {
  cl_uint index;
  std::vector<unsigned char> value;
};
struct M_driver_captured_dispatch {
  cl_kernel kernel;
  size_t global[3];
  size_t local[3];
  std::vector<M_driver_captured_arg> args;
};
// clSetKernelArg does not retain buffers. Keep native objects alive between
// handle validation and enqueue, without serializing blocking vendor calls
// across all queues with the context's bookkeeping mutex.
struct MemoryReferences {
  OpenCL &api;
  std::vector<cl_mem> objects{};
  cl_int retain(cl_mem object) {
    auto error = api.clRetainMemObject(object);
    if (!error)
      objects.push_back(object);
    return error;
  }
  ~MemoryReferences() {
    for (auto object : objects)
      api.clReleaseMemObject(object);
  }
};
// Readback storage outlives an event destroyed before its asynchronous copy.
// Dependencies are immutable snapshots, so re-recording an event cannot alter
// a previously enqueued cross-queue dependency.
struct DeviceErrorReadback {
  cl_uint code = 0;
  std::vector<std::shared_ptr<DeviceErrorReadback>> dependencies{};
};
struct OpenCLHostProfile {
  uint64_t launches = 0;
  uint64_t metadataCalls = 0;
  uint64_t setArgCalls = 0;
  uint64_t retainCalls = 0;
  uint64_t enqueueCalls = 0;
  uint64_t queueLockNs = 0;
  uint64_t executeNs = 0;
  uint64_t functionLockNs = 0;
  uint64_t metadataNs = 0;
  uint64_t setArgNs = 0;
  uint64_t retainNs = 0;
  uint64_t enqueueNs = 0;
  uint64_t flushCalls = 0;
  uint64_t flushNs = 0;
  uint64_t interDispatchGapNs = 0;
  uint64_t interDispatchGapMaxNs = 0;
  uint64_t interDispatchGapCount = 0;
  uint64_t copyFromNs = 0;
  uint64_t syncNs = 0;
  std::chrono::steady_clock::time_point lastDispatchEnd{};
};
struct OpenCLHostTimer {
  uint64_t *elapsed = nullptr;
  std::chrono::steady_clock::time_point start{};
  explicit OpenCLHostTimer(uint64_t *counter)
      : elapsed(counter), start(counter ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{}) {}
  ~OpenCLHostTimer() {
    if (elapsed)
      *elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  }
};
struct M_driver_queue {
  M_driver_context_t context;
  cl_command_queue native;
  cl_command_queue recordQueue = nullptr;
  OpenCL::Recording recording = nullptr;
  bool recordingReady = false;
  bool recordingDisabled = false;
  size_t recordableMaxSize = 4096;
  std::vector<M_driver_captured_dispatch> recordedDispatches{};
  std::vector<M_driver_captured_dispatch> pendingDispatches{};
  size_t pendingDispatchCount = 0;
  uint64_t dispatchCount = 0;
  uint64_t syncCount = 0;
  uint64_t commandCount = 0;
  bool profiling = false;
  bool hostProfiling = false;
  bool flushAfterEnqueue = false;
  OpenCLHostProfile hostProfile{};
  struct Sample {
    std::string name;
    cl_event event;
  };
  std::mutex profileMutex{};
  std::vector<Sample> samples{};
  uint64_t droppedSamples = 0;
  std::mutex errorMutex{};
  cl_mem deviceError = nullptr;
  bool failed = false;
  cl_event lastErrorEvent = nullptr;
  std::shared_ptr<DeviceErrorReadback> lastError{};
  std::vector<std::shared_ptr<DeviceErrorReadback>> dependencies{};
};
struct OpenCLDispatchEnd {
  M_driver_queue_t queue;
  ~OpenCLDispatchEnd() {
    if (queue->hostProfiling)
      queue->hostProfile.lastDispatchEnd = std::chrono::steady_clock::now();
  }
};
// Diagnostics use device timestamps without inserting per-kernel
// synchronization. Limit retained events so accidental long profiling runs
// cannot grow unbounded.
static void reportProfile(M_driver_queue_t q) {
  std::lock_guard<std::mutex> lock(q->profileMutex);
  if (q->samples.empty())
    return;
  auto &api = *q->context->device->api;
  struct Totals {
    uint64_t count = 0, ns = 0;
  };
  struct GapTotals {
    uint64_t count = 0, ns = 0, maxNs = 0;
    uint64_t lateQueueNs = 0, submitDelayNs = 0, deviceWaitNs = 0;
  };
  std::map<std::string, Totals> totals;
  std::map<std::pair<std::string, std::string>, GapTotals> gaps;
  auto label = [](const std::string &name) {
    auto marker = name.find("closure$");
    if (marker != std::string::npos) {
      auto begin = marker + sizeof("closure$") - 1;
      auto end = name.find("::execute", begin);
      if (end != std::string::npos)
        return name.substr(begin, end - begin);
    }
    marker = name.find("kernels::");
    if (marker != std::string::npos) {
      auto begin = marker + sizeof("kernels::") - 1;
      auto end = name.find('(', begin);
      if (end != std::string::npos)
        return name.substr(begin, end - begin);
    }
    return name.substr(0, 96);
  };
  uint64_t firstQueued = UINT64_MAX, firstStart = UINT64_MAX;
  uint64_t last = 0, sum = 0, failed = 0;
  uint64_t queuedToSubmit = 0, submitToStart = 0;
  uint64_t previousEnd = 0;
  std::string previousName;
  for (auto &sample : q->samples) {
    cl_ulong queued = 0, submit = 0, start = 0, end = 0;
    auto a = api.clGetEventProfilingInfo(sample.event,
                                         CL_PROFILING_COMMAND_QUEUED,
                                         sizeof(queued), &queued, nullptr);
    auto b = api.clGetEventProfilingInfo(sample.event,
                                         CL_PROFILING_COMMAND_SUBMIT,
                                         sizeof(submit), &submit, nullptr);
    auto c = api.clGetEventProfilingInfo(sample.event,
                                         CL_PROFILING_COMMAND_START,
                                         sizeof(start), &start, nullptr);
    auto d = api.clGetEventProfilingInfo(sample.event, CL_PROFILING_COMMAND_END,
                                         sizeof(end), &end, nullptr);
    std::fprintf(stderr, "EVENT_RAW %s %llu %llu %llu %llu %d %d %d %d\n", label(sample.name).c_str(), (unsigned long long)queued, (unsigned long long)submit, (unsigned long long)start, (unsigned long long)end, a,b,c,d);
    if (a == CL_SUCCESS && b == CL_SUCCESS && c == CL_SUCCESS &&
        d == CL_SUCCESS && queued <= submit && submit <= start &&
        end >= start) {
      auto &t = totals[sample.name];
      ++t.count;
      t.ns += end - start;
      firstQueued = std::min(firstQueued, uint64_t(queued));
      firstStart = std::min(firstStart, uint64_t(start));
      last = std::max(last, uint64_t(end));
      if (!previousName.empty() && start >= previousEnd) {
        auto gapNs = uint64_t(start - previousEnd);
        auto &gap = gaps[{previousName, label(sample.name)}];
        ++gap.count;
        gap.ns += gapNs;
        gap.maxNs = std::max(gap.maxNs, gapNs);
        uint64_t lateQueue = queued > previousEnd ? queued - previousEnd : 0;
        uint64_t submitBase = std::max(uint64_t(queued), previousEnd);
        uint64_t submitDelay = submit > submitBase ? submit - submitBase : 0;
        uint64_t deviceBase = std::max(uint64_t(submit), previousEnd);
        uint64_t deviceWait = start > deviceBase ? start - deviceBase : 0;
        gap.lateQueueNs += lateQueue;
        gap.submitDelayNs += submitDelay;
        gap.deviceWaitNs += deviceWait;
      }
      previousEnd = end;
      previousName = label(sample.name);
      sum += end - start;
      queuedToSubmit += submit - queued;
      submitToStart += start - submit;
    } else
      ++failed;
    api.clReleaseEvent(sample.event);
  }
  std::fprintf(stderr,
               "GPU_PROFILE samples=%zu dropped=%llu failed=%llu span_ms=%.3f "
               "kernel_ms=%.3f queue_to_submit_ms=%.3f "
               "submit_to_start_ms=%.3f queued_span_ms=%.3f\n",
               q->samples.size(), (unsigned long long)q->droppedSamples,
               (unsigned long long)failed,
               last > firstStart ? (last - firstStart) / 1e6 : 0.,
               sum / 1e6, queuedToSubmit / 1e6, submitToStart / 1e6,
               last > firstQueued ? (last - firstQueued) / 1e6 : 0.);
  std::vector<std::pair<std::pair<std::string, std::string>, GapTotals>>
      largestGaps(gaps.begin(), gaps.end());
  std::sort(largestGaps.begin(), largestGaps.end(), [](const auto &a,
                                                       const auto &b) {
    return a.second.ns > b.second.ns;
  });
  for (size_t i = 0; i < largestGaps.size(); ++i) {
    const auto &[transition, gap] = largestGaps[i];
    std::fprintf(stderr,
                 "GPU_GAP from=%s to=%s count=%llu total_ms=%.3f "
                 "max_us=%.3f late_queue_ms=%.3f submit_delay_ms=%.3f "
                 "device_wait_ms=%.3f\n",
                 transition.first.c_str(), transition.second.c_str(),
                 (unsigned long long)gap.count, gap.ns / 1e6,
                 gap.maxNs / 1e3, gap.lateQueueNs / 1e6,
                 gap.submitDelayNs / 1e6, gap.deviceWaitNs / 1e6);
  }
  for (auto &entry : totals)
    std::fprintf(stderr, "GPU_PROFILE %s calls=%llu ms=%.3f\n",
                 entry.first.c_str(), (unsigned long long)entry.second.count,
                 entry.second.ns / 1e6);
  q->samples.clear();
  q->droppedSamples = 0;
}
struct M_driver_runtime_bundle {
  M_driver_context_t context;
  cl_program native;
  std::vector<M::OpenCL::KernelABI> kernels{};
  bool strictABI = false;
};
struct M_driver_function {
  M_driver_context_t context;
  cl_kernel native;
  std::mutex mutex; // clSetKernelArg mutates the kernel object.
  std::vector<M::OpenCL::ArgumentABI> arguments{};
  bool strictABI = false;
  std::string name;
  bool dynamicShared = false;
  bool deviceErrors = false;
  uint32_t subgroupSize = 0;
};
struct M_driver_event {
  M_driver_context_t context;
  cl_event native = nullptr;
  std::shared_ptr<DeviceErrorReadback> failure{};
};
static bool sameDispatches(
    const std::vector<M_driver_captured_dispatch> &,
    const std::vector<M_driver_captured_dispatch> &, size_t);
static cl_int enqueueCapturedNormally(
    OpenCL &, cl_command_queue,
    const std::vector<M_driver_captured_dispatch> &, size_t);
static cl_int recordDispatches(M_driver_queue_t);
static cl_int flushRecordedDispatches(M_driver_queue_t);
static std::string &lastError() {
  static thread_local std::string message;
  return message;
}
static M_driver_status_t result(cl_int error) {
  if (error == CL_SUCCESS)
    return M_DRIVER_STATUS_SUCCESS;
  lastError() = "OpenCL error " + std::to_string(error);
  return M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1;
}
static constexpr M_driver_status_t deviceAssertionStatus =
    M_DRIVER_STATUS_CUSTOM_CODES_BASE - 2;
static bool readbackFailed(const std::shared_ptr<DeviceErrorReadback> &root) {
  std::vector<const DeviceErrorReadback *> work{root.get()};
  std::set<const DeviceErrorReadback *> visited;
  while (!work.empty()) {
    auto *value = work.back();
    work.pop_back();
    if (!value || !visited.insert(value).second)
      continue;
    if (value->code)
      return true;
    for (const auto &dependency : value->dependencies)
      work.push_back(dependency.get());
  }
  return false;
}
// Caller holds errorMutex. Polling folds completed snapshots into the sticky
// queue status and bounds the dependency chain when the device catches up.
static cl_int updateQueueError(M_driver_queue_t q) {
  if (!q->lastErrorEvent)
    return CL_SUCCESS;
  auto &api = *q->context->device->api;
  cl_int state;
  auto error =
      api.clGetEventInfo(q->lastErrorEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(state), &state, nullptr);
  if (error || state < 0)
    return error ? error : state;
  if (state == CL_COMPLETE) {
    q->failed |= readbackFailed(q->lastError);
    api.clReleaseEvent(q->lastErrorEvent);
    q->lastErrorEvent = nullptr;
    q->lastError.reset();
  }
  return CL_SUCCESS;
}
// The caller holds errorMutex and has completed all preceding queue commands.
static M_driver_status_t completed(M_driver_queue_t q, cl_int error) {
  if (error)
    return result(error);
  auto &api = *q->context->device->api;
  error = updateQueueError(q);
  if (error)
    return result(error);
  for (const auto &dependency : q->dependencies)
    q->failed |= readbackFailed(dependency);
  q->dependencies.clear();
  if (q->deviceError && !q->failed) {
    cl_uint code = 0;
    error = api.clEnqueueReadBuffer(q->native, q->deviceError, CL_TRUE, 0,
                                    sizeof(code), &code, 0, nullptr, nullptr);
    if (error)
      return result(error);
    q->failed |= code != 0;
  }
  return q->failed ? deviceAssertionStatus : M_DRIVER_STATUS_SUCCESS;
}
static void CL_CALLBACK releaseErrorReadback(cl_event, cl_int, void *data) {
  delete static_cast<std::shared_ptr<DeviceErrorReadback> *>(data);
}
static bool view(M_driver_context_t ctx, M_driver_memory_view_t v,
                 bool allowHost = false) {
  // The caller holds memoryMutex during validation, then retains the native
  // object before releasing that mutex for the vendor call.
  return ctx && v && ctx->memories.count(v->memory) &&
         (allowHost || !v->memory->hostAddress) &&
         v->offset <= v->memory->size && v->size <= v->memory->size - v->offset;
}
M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *output) {
  if (!version || !output)
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = nullptr;
  const auto major = version->major;
  *version = {M_DRIVER_INTERFACE_VERSION_MAJOR,
              M_DRIVER_INTERFACE_VERSION_MINOR,
              M_DRIVER_INTERFACE_VERSION_PATCH};
  if (major != version->major)
    return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;
  auto driver = std::make_unique<M_driver_driver>();
  if (!driver->api.load()) {
    const char *message = dlerror();
    lastError() = message ? message : "Unable to load OpenCL";
    return M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1;
  }
  cl_uint count = 0;
  auto error = driver->api.clGetPlatformIDs(0, nullptr, &count);
  if (error != CL_SUCCESS)
    return result(error);
  if (!count)
    return M_DRIVER_STATUS_NO_DEVICES;
  std::vector<cl_platform_id> platforms(count);
  if ((error = driver->api.clGetPlatformIDs(count, platforms.data(), nullptr)))
    return result(error);
  for (auto platform : platforms) {
    cl_uint n = 0;
    error = driver->api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                                       &n);
    if (error == CL_DEVICE_NOT_FOUND)
      continue;
    if (error)
      return result(error);
    std::vector<cl_device_id> devices(n);
    if ((error = driver->api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n,
                                            devices.data(), nullptr)))
      return result(error);
    for (auto device : devices)
      driver->devices.push_back(
          {&driver->api, device, int32_t(driver->devices.size())});
  }
  if (driver->devices.empty())
    return M_DRIVER_STATUS_NO_DEVICES;
  *output = driver.release();
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_destroy(M_driver_driver_t driver) {
  if (!driver)
    return M_DRIVER_STATUS_BAD_DRIVER;
  delete driver;
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_status_message(M_driver_driver_t,
                                          M_driver_status_t status,
                                          char *buffer, int64_t size) {
  if (!buffer || size <= 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (status != deviceAssertionStatus &&
      status != M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1 &&
      (status > 0 || status <= -M_DRIVER_NUM_STANDARD_STATUS_CODES))
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  std::string message = status == deviceAssertionStatus
                            ? "OpenCL device assertion failed"
                        : status == M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1
                            ? lastError()
                            : "HAL status " + std::to_string(status);
  std::snprintf(buffer, size_t(size), "%s", message.c_str());
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_property(M_driver_driver_t driver, const char *name,
                                    void *value) {
  if (!driver || !name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!std::strcmp(name, M_DRIVER_PROPERTY_API))
    *static_cast<const char **>(value) = "OpenCL";
  else if (!std::strcmp(name, M_DRIVER_PROPERTY_VERSION))
    *static_cast<M_driver_version *>(value) = {
        M_DRIVER_INTERFACE_VERSION_MAJOR, M_DRIVER_INTERFACE_VERSION_MINOR,
        M_DRIVER_INTERFACE_VERSION_PATCH};
  else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  if (!driver || !count)
    return M_DRIVER_STATUS_INVALID_ARG;
  *count = driver->devices.size();
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_device_get(M_driver_driver_t driver, int64_t id,
                                      M_driver_device_t *device) {
  if (!driver || !device)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (id < 0 || uint64_t(id) >= driver->devices.size())
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = &driver->devices[id];
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_device_property(M_driver_device_t d,
                                           const char *name, void *value) {
  if (!d || !name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!std::strcmp(name, M_DRIVER_DEVICE_PROPERTY_ARCH) ||
      !std::strcmp(name, M_DRIVER_DEVICE_PROPERTY_NAME)) {
    size_t size = 0;
    auto error =
        d->api->clGetDeviceInfo(d->native, CL_DEVICE_NAME, 0, nullptr, &size);
    if (error)
      return result(error);
    std::vector<char> text(size ? size : 1);
    error = d->api->clGetDeviceInfo(d->native, CL_DEVICE_NAME, text.size(),
                                    text.data(), nullptr);
    if (error)
      return result(error);
    const char *description = text.data();
    if (!std::strcmp(name, M_DRIVER_DEVICE_PROPERTY_ARCH))
      description = std::strcmp(text.data(), "QUALCOMM Adreno(TM) 840") == 0
                        ? "adreno-840"
                        : "opencl-generic";
    std::snprintf(static_cast<char *>(value), M_DRIVER_DEVICE_PROPERTY_MAX_LEN,
                  "%s", description);
  } else if (!std::strcmp(name, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE)) {
    // DLPack kDLOpenCL = 4; this is not a CUDA device pointer.
    *static_cast<M_driver_dlpack_device *>(value) = {4, d->ordinal};
  } else if (!std::strcmp(name, M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED)) {
    cl_bool unified = CL_FALSE;
    auto error =
        d->api->clGetDeviceInfo(d->native, CL_DEVICE_HOST_UNIFIED_MEMORY,
                                sizeof(unified), &unified, nullptr);
    if (error)
      return result(error);
    *static_cast<int64_t *>(value) = unified != CL_FALSE;
  } else if (!std::strcmp(name, M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE)) {
    cl_ulong size = 0;
    auto error = d->api->clGetDeviceInfo(
        d->native, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size), &size, nullptr);
    if (error)
      return result(error);
    *static_cast<uint64_t *>(value) = size;
  } else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_context_create(M_driver_device_t d,
                                          M_driver_context_t *output) {
  if (!d || !output)
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = nullptr;
  cl_int error = CL_SUCCESS;
  cl_context native =
      d->api->clCreateContext(nullptr, 1, &d->native, nullptr, nullptr, &error);
  if (error)
    return result(error);
  auto queue = d->api->clCreateCommandQueue(native, d->native, 0, &error);
  if (error) {
    d->api->clReleaseContext(native);
    return result(error);
  }
  *output = new M_driver_context{d, native, queue};
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_context_destroy(M_driver_context_t ctx) {
  if (!ctx)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  auto &api = *ctx->device->api;
  auto error = api.clFinish(ctx->transfer);
  api.clReleaseCommandQueue(ctx->transfer);
  api.clReleaseContext(ctx->native);
  delete ctx;
  return result(error);
}
// Pinned allocations remain mapped for host access. They are staging storage,
// not device views: copies use the mapped host address and device allocations
// stay opaque. Unmap before releasing the native allocation.
M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t ctx,
                                               uint64_t size,
                                               M_driver_memory_t *output) {
  if (!ctx || !output || size > SIZE_MAX)
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = nullptr;
  auto &api = *ctx->device->api;
  cl_int error;
  auto bytes = std::max(size_t(size), size_t(1));
  auto native =
      api.clCreateBuffer(ctx->native, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         bytes, nullptr, &error);
  if (error)
    return result(error);
  auto address = api.clEnqueueMapBuffer(ctx->transfer, native, CL_TRUE,
                                        CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0,
                                        nullptr, nullptr, &error);
  if (error) {
    api.clReleaseMemObject(native);
    return result(error);
  }
  *output = new M_driver_memory{ctx, native, size, address};
  if (std::getenv("MODULAR_OPENCL_DEBUG"))
    std::fprintf(stderr, "OPENCL_DEBUG ALLOC_PINNED ctx=%p mem=%p size=%llu\n",
                 static_cast<void *>(ctx), static_cast<void *>(*output),
                 (unsigned long long)size);
  std::lock_guard<std::mutex> lock(ctx->memoryMutex);
  ctx->memories.insert(*output);
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t ctx,
                                              M_driver_memory_t mem) {
  if (!ctx || !mem)
    return M_DRIVER_STATUS_BAD_MEMORY;
  std::lock_guard<std::mutex> lock(ctx->memoryMutex);
  if (!ctx->memories.count(mem) || !mem->hostAddress)
    return M_DRIVER_STATUS_BAD_MEMORY;
  auto &api = *ctx->device->api;
  auto error = api.clEnqueueUnmapMemObject(
      ctx->transfer, mem->native, mem->hostAddress, 0, nullptr, nullptr);
  if (error)
    return result(error);
  error = api.clFinish(ctx->transfer);
  auto releaseError = api.clReleaseMemObject(mem->native);
  ctx->memories.erase(mem);
  delete mem;
  return result(error ? error : releaseError);
}
M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t ctx,
                                             uint64_t size,
                                             M_driver_memory_t *output) {
  if (!ctx || !output || size > SIZE_MAX)
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = nullptr;
  cl_int error;
  auto mem = ctx->device->api->clCreateBuffer(ctx->native, CL_MEM_READ_WRITE,
                                              std::max(size_t(size), size_t(1)),
                                              nullptr, &error);
  if (error)
    return result(error);
  *output = new M_driver_memory{ctx, mem, size};
  if (std::getenv("MODULAR_OPENCL_DEBUG"))
    std::fprintf(stderr, "OPENCL_DEBUG ALLOC_SYNC ctx=%p mem=%p size=%llu\n",
                 static_cast<void *>(ctx), static_cast<void *>(*output),
                 (unsigned long long)size);
  std::lock_guard<std::mutex> lock(ctx->memoryMutex);
  ctx->memories.insert(*output);
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_memory_free_sync(M_driver_context_t ctx,
                                            M_driver_memory_t mem) {
  if (std::getenv("MODULAR_OPENCL_DEBUG")) {
    std::fprintf(stderr, "OPENCL_DEBUG FREE_SYNC ctx=%p mem=%p\n",
                 static_cast<void *>(ctx), static_cast<void *>(mem));
    void *frames[24];
    auto depth = backtrace(frames, 24);
    backtrace_symbols_fd(frames, depth, 2);
  }
  if (!ctx || !mem)
    return M_DRIVER_STATUS_BAD_MEMORY;
  std::lock_guard<std::mutex> lock(ctx->memoryMutex);
  if (!ctx->memories.count(mem) || mem->hostAddress)
    return M_DRIVER_STATUS_BAD_MEMORY;
  ctx->memories.erase(mem);
  auto error = ctx->device->api->clReleaseMemObject(mem->native);
  delete mem;
  return result(error);
}
M_driver_status_t M_driver_memory_property(M_driver_memory_t mem,
                                           const char *name, void *value) {
  if (!mem || !name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!std::strcmp(name, M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS)) {
    if (!mem->hostAddress)
      return M_DRIVER_STATUS_UNSUPPORTED;
    *static_cast<uint64_t *>(value) =
        reinterpret_cast<uintptr_t>(mem->hostAddress);
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (std::strcmp(name, M_DRIVER_MEMORY_PROPERTY_ADDRESS))
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  *static_cast<uint64_t *>(value) = reinterpret_cast<uintptr_t>(mem->native);
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_queue_create(M_driver_context_t ctx,
                                        M_driver_queue_t *output) {
  if (!ctx || !output)
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = nullptr;
  cl_int error;
  const char *profile = std::getenv("MODULAR_OPENCL_PROFILE");
  bool profiling = profile && std::strcmp(profile, "1") == 0;
  auto q = ctx->device->api->clCreateCommandQueue(
      ctx->native, ctx->device->native,
      profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &error);
  if (error)
    return result(error);
  *output = new M_driver_queue{ctx, q};
  (*output)->profiling = profiling;
  const char *hostProfile = std::getenv("MODULAR_OPENCL_HOST_PROFILE");
  (*output)->hostProfiling = hostProfile && !std::strcmp(hostProfile, "1");
  const char *flushAfterEnqueue =
      std::getenv("MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE");
  (*output)->flushAfterEnqueue =
      flushAfterEnqueue && !std::strcmp(flushAfterEnqueue, "1");
  auto &api = *ctx->device->api;
  const char *recording = std::getenv("MODULAR_OPENCL_RECORDING");
  if (recording && !std::strcmp(recording, "1") && api.newRecording &&
      api.endRecording && api.releaseRecording && api.enqueueRecording &&
      !profiling) {
    (*output)->recordQueue = api.clCreateCommandQueue(
        ctx->native, ctx->device->native, cl_command_queue_properties(1u << 30),
        &error);
    if (error || !(*output)->recordQueue) {
      (*output)->recordingDisabled = true;
    } else {
      cl_uint maxSize = 0;
      if (!api.clGetDeviceInfo(ctx->device->native, 0x41DE, sizeof(maxSize),
                               &maxSize, nullptr) && maxSize)
        (*output)->recordableMaxSize = maxSize;
      if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
        std::fprintf(stderr, "OPENCL_RECORDING QCOM recordable queue enabled\n");
    }
  }
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t q) {
  if (!q)
    return M_DRIVER_STATUS_BAD_QUEUE;
  std::lock_guard<std::mutex> guard(q->errorMutex);
  auto &api = *q->context->device->api;
  auto syncStart = std::chrono::steady_clock::now();
  auto error = flushRecordedDispatches(q);
  if (!error)
    error = api.clFinish(q->native);
  ++q->syncCount;
  if (error == CL_SUCCESS && q->profiling)
    reportProfile(q);
  if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
    std::fprintf(stderr,
                 "OPENCL_QUEUE_DEBUG queue=%p sync=%llu dispatches=%llu "
                 "other_commands=%llu\n",
                 static_cast<void *>(q), (unsigned long long)q->syncCount,
                 (unsigned long long)q->dispatchCount,
                 (unsigned long long)q->commandCount);
  auto status = completed(q, error);
  if (q->hostProfiling) {
    q->hostProfile.syncNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - syncStart)
            .count();
    const auto &p = q->hostProfile;
    const uint64_t nested = p.functionLockNs + p.metadataNs + p.setArgNs +
                            p.retainNs + p.enqueueNs + p.flushNs;
    const uint64_t otherNs = p.executeNs > nested ? p.executeNs - nested : 0;
    std::fprintf(stderr,
                 "OPENCL_HOST_PROFILE launches=%llu metadata_calls=%llu "
                 "set_arg_calls=%llu retain_calls=%llu enqueue_calls=%llu "
                 "queue_lock_ms=%.3f "
                 "launch_ms=%.3f function_lock_ms=%.3f metadata_ms=%.3f "
                 "set_arg_ms=%.3f retain_ms=%.3f enqueue_ms=%.3f "
                 "flush_calls=%llu flush_ms=%.3f "
                 "launch_other_ms=%.3f inter_dispatch_gap_ms=%.3f "
                 "inter_dispatch_gap_max_us=%.3f inter_dispatch_gap_count=%llu "
                 "copy_from_ms=%.3f sync_ms=%.3f\n",
                 (unsigned long long)p.launches,
                 (unsigned long long)p.metadataCalls,
                 (unsigned long long)p.setArgCalls,
                 (unsigned long long)p.retainCalls,
                 (unsigned long long)p.enqueueCalls, p.queueLockNs / 1e6,
                 p.executeNs / 1e6, p.functionLockNs / 1e6,
                 p.metadataNs / 1e6, p.setArgNs / 1e6, p.retainNs / 1e6,
                 p.enqueueNs / 1e6, (unsigned long long)p.flushCalls,
                 p.flushNs / 1e6, otherNs / 1e6,
                 p.interDispatchGapNs / 1e6,
                 p.interDispatchGapMaxNs / 1e3,
                 (unsigned long long)p.interDispatchGapCount,
                 p.copyFromNs / 1e6,
                 p.syncNs / 1e6);
    q->hostProfile = {};
  }
  return status;
}
M_driver_status_t M_driver_queue_destroy(M_driver_context_t ctx,
                                         M_driver_queue_t q) {
  if (!q || q->context != ctx)
    return M_DRIVER_STATUS_BAD_QUEUE;
  auto &api = *ctx->device->api;
  // Drain readbacks before destroying their queue-owned storage. Completion
  // callbacks retain snapshots independently of both the queue and SDK event.
  auto error = api.clFinish(q->native);
  auto status = completed(q, error);
  if (q->profiling)
    reportProfile(q);
  if (q->lastErrorEvent)
    api.clReleaseEvent(q->lastErrorEvent);
  if (q->deviceError)
    api.clReleaseMemObject(q->deviceError);
  if (q->recording)
    api.releaseRecording(q->recording);
  if (q->recordQueue)
    api.clReleaseCommandQueue(q->recordQueue);
  api.clReleaseCommandQueue(q->native);
  delete q;
  return status;
}
M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t q, bool *value) {
  if (!q || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  *value = true;
  return M_DRIVER_STATUS_SUCCESS;
}
static bool sameDispatches(
    const std::vector<M_driver_captured_dispatch> &a,
    const std::vector<M_driver_captured_dispatch> &b, size_t bCount) {
  if (a.size() != bCount || b.size() < bCount)
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].kernel != b[i].kernel ||
        std::memcmp(a[i].global, b[i].global, sizeof(a[i].global)) ||
        std::memcmp(a[i].local, b[i].local, sizeof(a[i].local)) ||
        a[i].args.size() != b[i].args.size()) {
      if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
        std::fprintf(stderr,
                     "OPENCL_RECORDING mismatch dispatch=%zu kernel=%d "
                     "global=%d local=%d args=%zu/%zu\n",
                     i, a[i].kernel == b[i].kernel,
                     !std::memcmp(a[i].global, b[i].global, sizeof(a[i].global)),
                     !std::memcmp(a[i].local, b[i].local, sizeof(a[i].local)),
                     a[i].args.size(), b[i].args.size());
      return false;
    }
    for (size_t j = 0; j < a[i].args.size(); ++j)
      if (a[i].args[j].index != b[i].args[j].index ||
          a[i].args[j].value.size() != b[i].args[j].value.size()) {
        if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
          std::fprintf(stderr,
                       "OPENCL_RECORDING mismatch dispatch=%zu arg=%zu "
                       "index=%u/%u size=%zu/%zu\n",
                       i, j, a[i].args[j].index, b[i].args[j].index,
                       a[i].args[j].value.size(), b[i].args[j].value.size());
        return false;
      }
  }
  return true;
}
static cl_int enqueueCapturedNormally(
    OpenCL &api, cl_command_queue queue,
    const std::vector<M_driver_captured_dispatch> &dispatches,
    size_t dispatchCount) {
  for (size_t i = 0; i < dispatchCount; ++i) {
    const auto &dispatch = dispatches[i];
    for (const auto &arg : dispatch.args) {
      auto error = api.clSetKernelArg(dispatch.kernel, arg.index,
                                      arg.value.size(), arg.value.data());
      if (error)
        return error;
    }
    auto error = api.clEnqueueNDRangeKernel(queue, dispatch.kernel, 3, nullptr,
                                            dispatch.global, dispatch.local, 0,
                                            nullptr, nullptr);
    if (error)
      return error;
  }
  return CL_SUCCESS;
}
static cl_int recordDispatches(M_driver_queue_t q) {
  auto &api = *q->context->device->api;
  cl_int error = CL_SUCCESS;
  q->recording = api.newRecording(q->recordQueue, &error);
  if (error || !q->recording)
    return error ? error : CL_INVALID_OPERATION;
  size_t index = 0;
  for (size_t dispatchIndex = 0; dispatchIndex < q->pendingDispatchCount;
       ++dispatchIndex) {
    const auto &dispatch = q->pendingDispatches[dispatchIndex];
    for (const auto &arg : dispatch.args) {
      error = api.clSetKernelArg(dispatch.kernel, arg.index, arg.value.size(),
                                 arg.value.data());
      if (error)
        return error;
    }
    error = api.clEnqueueNDRangeKernel(q->recordQueue, dispatch.kernel, 3,
                                       nullptr, dispatch.global,
                                       dispatch.local, 0, nullptr, nullptr);
    if (error) {
      if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
        std::fprintf(stderr,
                     "OPENCL_RECORDING enqueue_error=%d dispatch=%zu\n",
                     error, index);
      return error;
    }
    ++index;
  }
  error = api.endRecording(q->recording);
  if (!error)
    error = api.clFinish(q->recordQueue);
  if (!error)
    q->recordingReady = true;
  return error;
}
static cl_int flushRecordedDispatches(M_driver_queue_t q) {
  if (!q->pendingDispatchCount)
    return CL_SUCCESS;
  auto &api = *q->context->device->api;
  cl_int error = CL_SUCCESS;
  if (q->recordingReady) {
    if (sameDispatches(q->recordedDispatches, q->pendingDispatches,
                       q->pendingDispatchCount)) {
      std::vector<OpenCL::ArrayArg> args;
      for (size_t dispatch = 0; dispatch < q->pendingDispatchCount;
           ++dispatch)
        for (size_t index = 0;
             index < q->pendingDispatches[dispatch].args.size(); ++index) {
          const auto &arg = q->pendingDispatches[dispatch].args[index];
          const auto &original = q->recordedDispatches[dispatch].args[index];
          if (arg.value != original.value)
            args.push_back({cl_uint(dispatch), arg.index, arg.value.size(),
                            arg.value.data()});
        }
      error = api.enqueueRecording(q->native, q->recording, args.size(),
                                   args.data(), 0, nullptr, 0, nullptr, 0,
                                   nullptr, 0, nullptr, nullptr);
      // Qualcomm mutable updates persist on the recording. Compare the next
      // token against the arguments most recently applied, not the initial
      // capture, and advance that state only after a successful enqueue.
      if (!error)
        for (size_t dispatch = 0; dispatch < q->pendingDispatchCount;
             ++dispatch)
          for (size_t index = 0;
               index < q->pendingDispatches[dispatch].args.size(); ++index)
            q->recordedDispatches[dispatch].args[index].value =
                q->pendingDispatches[dispatch].args[index].value;
      if (!error && std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
        std::fprintf(stderr,
                     "OPENCL_RECORDING replayed dispatches=%zu args=%zu\n",
                     q->pendingDispatchCount, args.size());
      if (error) {
        auto replayError = error;
        error = enqueueCapturedNormally(api, q->native, q->pendingDispatches,
                                        q->pendingDispatchCount);
        q->recordingDisabled = true;
        api.releaseRecording(q->recording);
        q->recording = nullptr;
        q->recordingReady = false;
        if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
          std::fprintf(stderr,
                       "OPENCL_RECORDING replay_error=%d fallback_error=%d\n",
                       replayError, error);
      }
    } else {
      const size_t expectedDispatchCount = q->recordedDispatches.size();
      error = enqueueCapturedNormally(api, q->native, q->pendingDispatches,
                                      q->pendingDispatchCount);
      // The decode command sequence can change when a kernel selects a new
      // shape. Run this batch normally, then record the new sequence so the
      // following tokens can still use replay.
      api.releaseRecording(q->recording);
      q->recording = nullptr;
      q->recordingReady = false;
      bool recaptured = false;
      if (!error && q->recordQueue &&
          q->pendingDispatchCount <= q->recordableMaxSize) {
        auto captureError = recordDispatches(q);
        if (!captureError) {
          recaptured = true;
          q->recordedDispatches.assign(q->pendingDispatches.begin(),
                                       q->pendingDispatches.begin() +
                                           q->pendingDispatchCount);
        } else {
          q->recordingDisabled = true;
          if (q->recording) {
            api.releaseRecording(q->recording);
            q->recording = nullptr;
          }
        }
      } else if (!error) {
        q->recordingDisabled = true;
      }
      if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
        std::fprintf(stderr,
                     "OPENCL_RECORDING sequence_changed expected=%zu "
                     "actual=%zu recaptured=%d\n",
                     expectedDispatchCount, q->pendingDispatchCount,
                     recaptured);
    }
  } else {
    error = api.clFinish(q->native);
    if (!error && q->recordQueue && !q->recordingDisabled &&
        q->pendingDispatchCount >= 64 &&
        q->pendingDispatchCount <= q->recordableMaxSize) {
      error = recordDispatches(q);
      if (!error) {
        q->recordedDispatches.assign(q->pendingDispatches.begin(),
                                     q->pendingDispatches.begin() +
                                         q->pendingDispatchCount);
        if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
          std::fprintf(stderr, "OPENCL_RECORDING captured dispatches=%zu\n",
                       q->recordedDispatches.size());
      } else {
        q->recordingDisabled = true;
        if (q->recording) {
          api.releaseRecording(q->recording);
          q->recording = nullptr;
        }
        if (std::getenv("MODULAR_OPENCL_RECORDING_DEBUG"))
          std::fprintf(stderr, "OPENCL_RECORDING disabled capture_error=%d\n",
                       error);
        // The ordinary queue already executed this batch successfully.
        error = CL_SUCCESS;
      }
    } else if (!error && q->recordQueue && !q->recordingDisabled &&
               q->pendingDispatchCount > q->recordableMaxSize &&
               std::getenv("MODULAR_OPENCL_RECORDING_DEBUG")) {
      std::fprintf(stderr,
                   "OPENCL_RECORDING skipped oversized sequence=%zu limit=%zu\n",
                   q->pendingDispatchCount, q->recordableMaxSize);
    }
  }
  q->pendingDispatchCount = 0;
  return error;
}
M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t q,
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  if (!q)
    return M_DRIVER_STATUS_INVALID_ARG;
  std::lock_guard<std::mutex> queueGuard(q->errorMutex);
  auto &api = *q->context->device->api;
  MemoryReferences references{api};
  std::unique_lock<std::mutex> lock(q->context->memoryMutex);
  if (!view(q->context, dst, true))
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!dst->size)
    return M_DRIVER_STATUS_SUCCESS;
  if (!src)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (auto error = flushRecordedDispatches(q))
    return result(error);
  ++q->commandCount;
  if (dst->memory->hostAddress) {
    auto error = api.clFinish(q->native);
    if (!error)
      std::memmove(static_cast<char *>(dst->memory->hostAddress) + dst->offset,
                   src, dst->size);
    return completed(q, error);
  }
  auto native = dst->memory->native;
  auto error = references.retain(native);
  if (error)
    return result(error);
  lock.unlock();
  return completed(q, api.clEnqueueWriteBuffer(q->native, native, CL_TRUE,
                                               dst->offset, dst->size, src, 0,
                                               nullptr, nullptr));
}
M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t q, void *dst,
                                                  M_driver_memory_view_t src) {
  if (!q)
    return M_DRIVER_STATUS_INVALID_ARG;
  OpenCLHostTimer copyTimer(q->hostProfiling ? &q->hostProfile.copyFromNs
                                            : nullptr);
  std::lock_guard<std::mutex> queueGuard(q->errorMutex);
  auto &api = *q->context->device->api;
  MemoryReferences references{api};
  std::unique_lock<std::mutex> lock(q->context->memoryMutex);
  if (!view(q->context, src, true))
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!src->size)
    return M_DRIVER_STATUS_SUCCESS;
  if (!dst)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (auto error = flushRecordedDispatches(q))
    return result(error);
  ++q->commandCount;
  if (src->memory->hostAddress) {
    auto error = api.clFinish(q->native);
    if (!error)
      std::memmove(dst,
                   static_cast<char *>(src->memory->hostAddress) + src->offset,
                   src->size);
    return completed(q, error);
  }
  auto native = src->memory->native;
  auto error = references.retain(native);
  if (error)
    return result(error);
  lock.unlock();
  error = api.clEnqueueReadBuffer(q->native, native, CL_TRUE, src->offset,
                                  src->size, dst, 0, nullptr, nullptr);
  return completed(q, error);
}
M_driver_status_t M_driver_copy_to_device_sync(M_driver_context_t ctx,
                                               M_driver_memory_view_t dst,
                                               const void *src) {
  if (!ctx)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  M_driver_queue q{ctx, ctx->transfer};
  return M_driver_queue_copy_to_device(&q, dst, src);
}
M_driver_status_t M_driver_copy_from_device_sync(M_driver_context_t ctx,
                                                 void *dst,
                                                 M_driver_memory_view_t src) {
  if (!ctx)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  M_driver_queue q{ctx, ctx->transfer};
  return M_driver_queue_copy_from_device(&q, dst, src);
}
M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t q,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  if (!q)
    return M_DRIVER_STATUS_INVALID_ARG;
  std::lock_guard<std::mutex> queueGuard(q->errorMutex);
  auto &api = *q->context->device->api;
  MemoryReferences references{api};
  std::unique_lock<std::mutex> lock(q->context->memoryMutex);
  if (!view(q->context, dst, true) || !view(q->context, src, true) ||
      dst->size != src->size)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!src->size)
    return M_DRIVER_STATUS_SUCCESS;
  if (auto error = flushRecordedDispatches(q))
    return result(error);
  ++q->commandCount;
  auto hostSource =
      src->memory->hostAddress
          ? static_cast<char *>(src->memory->hostAddress) + src->offset
          : nullptr;
  auto hostDest =
      dst->memory->hostAddress
          ? static_cast<char *>(dst->memory->hostAddress) + dst->offset
          : nullptr;
  // Keep mapped storage alive under the bookkeeping lock until these blocking
  // transfers complete. Device-only copies below remain asynchronous.
  if (hostSource && hostDest) {
    auto error = api.clFinish(q->native);
    if (!error)
      std::memmove(hostDest, hostSource, src->size);
    return completed(q, error);
  }
  if (hostSource)
    return completed(q,
                     api.clEnqueueWriteBuffer(q->native, dst->memory->native,
                                              CL_TRUE, dst->offset, dst->size,
                                              hostSource, 0, nullptr, nullptr));
  if (hostDest)
    return completed(q, api.clEnqueueReadBuffer(q->native, src->memory->native,
                                                CL_TRUE, src->offset, src->size,
                                                hostDest, 0, nullptr, nullptr));
  auto nativeSource = src->memory->native, nativeDest = dst->memory->native;
  auto error = references.retain(nativeSource);
  if (!error)
    error = references.retain(nativeDest);
  if (error)
    return result(error);
  lock.unlock();
  return result(api.clEnqueueCopyBuffer(q->native, nativeSource, nativeDest,
                                        src->offset, dst->offset, src->size, 0,
                                        nullptr, nullptr));
}
M_driver_status_t M_driver_queue_fill(M_driver_queue_t q,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t size) {
  if (!q)
    return M_DRIVER_STATUS_INVALID_ARG;
  std::lock_guard<std::mutex> queueGuard(q->errorMutex);
  auto &api = *q->context->device->api;
  MemoryReferences references{api};
  std::unique_lock<std::mutex> lock(q->context->memoryMutex);
  if (!view(q->context, dst, true) ||
      !(size == 1 || size == 2 || size == 4 || size == 8) ||
      dst->offset % size || dst->size % size)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!dst->size)
    return M_DRIVER_STATUS_SUCCESS;
  if (auto error = flushRecordedDispatches(q))
    return result(error);
  ++q->commandCount;
  if (dst->memory->hostAddress) {
    auto error = api.clFinish(q->native);
    if (!error) {
      auto address =
          static_cast<char *>(dst->memory->hostAddress) + dst->offset;
      for (uint64_t offset = 0; offset < dst->size; offset += size)
        std::memcpy(address + offset, &value, size);
    }
    return completed(q, error);
  }
  auto native = dst->memory->native;
  auto error = references.retain(native);
  if (error)
    return result(error);
  lock.unlock();
  return result(api.clEnqueueFillBuffer(q->native, native, &value, size,
                                        dst->offset, dst->size, 0, nullptr,
                                        nullptr));
}
M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t q,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  return M_driver_queue_fill(q, dst, value, 1);
}
M_driver_status_t
M_driver_bundle_load(M_driver_context_t ctx, M_driver_static_bundle_t bundle,
                     M_driver_bundle_compilation_options_t opts,
                     M_driver_runtime_bundle_t *output) {
  if (!ctx || !bundle || !output)
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = nullptr;
  if (opts && (opts->debug_level_len || opts->optimization_level < -1 ||
               opts->optimization_level > 3))
    return M_DRIVER_STATUS_UNSUPPORTED;
  if (!bundle->file_type)
    return M_DRIVER_STATUS_INCOMPATIBLE;
  std::string fileType(bundle->file_type, bundle->file_type_len);
  bool strictABI = fileType == "mojo-opencl-v3";
  if (!strictABI && fileType != "opencl-source")
    return M_DRIVER_STATUS_INCOMPATIBLE;
  if (!bundle->mapped_data.data || !bundle->mapped_data.size)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  auto &api = *ctx->device->api;
  const char *source = bundle->mapped_data.data;
  size_t size = bundle->mapped_data.size;
  M::OpenCL::Bundle compiled;
  if (strictABI) {
    if (!M::OpenCL::decodeBundle(source, size, compiled))
      return M_DRIVER_STATUS_BAD_BUNDLE;
    cl_uint addressBits = 0;
    cl_bool littleEndian = CL_FALSE;
    auto status =
        api.clGetDeviceInfo(ctx->device->native, CL_DEVICE_ADDRESS_BITS,
                            sizeof(addressBits), &addressBits, nullptr);
    if (status)
      return result(status);
    status = api.clGetDeviceInfo(ctx->device->native, CL_DEVICE_ENDIAN_LITTLE,
                                 sizeof(littleEndian), &littleEndian, nullptr);
    if (status)
      return result(status);
    if (addressBits != 64 || !littleEndian)
      return M_DRIVER_STATUS_UNSUPPORTED;
    if (!compiled.requiredExtensions.empty()) {
      size_t length = 0;
      status = api.clGetDeviceInfo(ctx->device->native, CL_DEVICE_EXTENSIONS, 0,
                                   nullptr, &length);
      if (status)
        return result(status);
      std::vector<char> extensions(length + 1, 0);
      status = api.clGetDeviceInfo(ctx->device->native, CL_DEVICE_EXTENSIONS,
                                   length, extensions.data(), nullptr);
      if (status)
        return result(status);
      std::string supported = " " + std::string(extensions.data()) + " ";
      for (const auto &required : compiled.requiredExtensions)
        if (supported.find(" " + required + " ") == std::string::npos) {
          lastError() = "OpenCL device lacks required extension: " + required;
          return M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1;
        }
    }
    source = compiled.source.data();
    size = compiled.source.size();
  }
  const char *fastEnv = std::getenv("MODULAR_OPENCL_FAST_MATH");
  bool fastMath = fastEnv && std::strcmp(fastEnv, "1") == 0;
  std::string fastSource;
  if (fastMath) {
    fastSource.assign(source, size);
    const std::string strict = "FP_CONTRACT OFF";
    if (auto pos = fastSource.find(strict); pos != std::string::npos)
      fastSource.replace(pos, strict.size(), "FP_CONTRACT ON");
    source = fastSource.data();
    size = fastSource.size();
  }
  cl_int error;
  auto program =
      api.clCreateProgramWithSource(ctx->native, 1, &source, &size, &error);
  if (error)
    return result(error);
  std::string sourceText(source, size);
  bool modern =
      compiled.openclCVersion >= 300 ||
      sourceText.find("cl_khr_integer_dot_product") != std::string::npos ||
      sourceText.find("cl_khr_subgroups") != std::string::npos;
  std::string buildOptions = modern ? "-cl-std=CL3.0" : "-cl-std=CL1.2";
  if (opts && !opts->optimization_level)
    buildOptions += " -cl-opt-disable";
  else if (fastMath)
    buildOptions += " -cl-fast-relaxed-math -cl-mad-enable";
  const char *options = buildOptions.c_str();
  error = api.clBuildProgram(program, 1, &ctx->device->native, options, nullptr,
                             nullptr);
  if (error) {
    size_t n = 0;
    api.clGetProgramBuildInfo(program, ctx->device->native,
                              CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::vector<char> log(n + 1, 0);
    api.clGetProgramBuildInfo(program, ctx->device->native,
                              CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    result(error);
    lastError() += ": " + std::string(log.data());
    api.clReleaseProgram(program);
    return M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1;
  }
  *output = new M_driver_runtime_bundle{ctx, program};
  (*output)->strictABI = strictABI;
  (*output)->kernels = std::move(compiled.kernels);
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_bundle_unload(M_driver_context_t ctx,
                                         M_driver_runtime_bundle_t bundle) {
  if (!bundle || bundle->context != ctx)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  auto error = ctx->device->api->clReleaseProgram(bundle->native);
  delete bundle;
  return result(error);
}
M_driver_status_t M_driver_function_load(M_driver_context_t ctx,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *name, uint64_t size,
                                         M_driver_function_t *output) {
  if (!ctx || !bundle || bundle->context != ctx || !name || !size || !output)
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = nullptr;
  std::string symbol(name, size);
  const M::OpenCL::KernelABI *abi = nullptr;
  if (bundle->strictABI) {
    for (const auto &kernel : bundle->kernels)
      if (kernel.name == symbol)
        abi = &kernel;
    if (!abi)
      return M_DRIVER_STATUS_UNKNOWN_SYMBOL;
  }
  cl_int error;
  auto native = ctx->device->api->clCreateKernel(bundle->native,
                                                 abi && !abi->nativeName.empty()
                                                     ? abi->nativeName.c_str()
                                                     : symbol.c_str(),
                                                 &error);
  if (error)
    return result(error);
  if (abi) {
    cl_uint count = 0;
    error = ctx->device->api->clGetKernelInfo(native, CL_KERNEL_NUM_ARGS,
                                              sizeof(count), &count, nullptr);
    if (error || count != M::OpenCL::nativeArgumentCount(*abi)) {
      ctx->device->api->clReleaseKernel(native);
      return error ? result(error) : M_DRIVER_STATUS_BAD_BUNDLE;
    }
  }
  auto fn = new M_driver_function;
  fn->context = ctx;
  fn->native = native;
  fn->name = symbol;
  fn->strictABI = bundle->strictABI;
  if (abi)
    fn->arguments = abi->arguments;
  if (abi)
    fn->dynamicShared = abi->dynamicShared;
  if (abi)
    fn->deviceErrors = abi->deviceErrors;
  if (abi)
    fn->subgroupSize = abi->subgroupSize;
  *output = fn;
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_function_unload(M_driver_context_t ctx,
                                           M_driver_function_t fn) {
  if (!fn || fn->context != ctx)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  auto error = ctx->device->api->clReleaseKernel(fn->native);
  delete fn;
  return result(error);
}
M_driver_status_t
M_driver_queue_execute(M_driver_queue_t q, M_driver_function_t fn,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *sizes, uint32_t count) {
  if (!q || !fn || q->context != fn->context || !config ||
      (count && (!args || !sizes)))
    return M_DRIVER_STATUS_INVALID_ARG;
  const auto dispatchEntry = q->hostProfiling
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  auto queueWaitStart = q->hostProfiling ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
  std::lock_guard<std::mutex> queueGuard(q->errorMutex);
  if (q->hostProfiling) {
    q->hostProfile.queueLockNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - queueWaitStart)
            .count();
    if (q->hostProfile.lastDispatchEnd !=
        std::chrono::steady_clock::time_point{}) {
      auto gap = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     dispatchEntry - q->hostProfile.lastDispatchEnd)
                     .count();
      if (gap > 0) {
        q->hostProfile.interDispatchGapNs += uint64_t(gap);
        q->hostProfile.interDispatchGapMaxNs =
            std::max(q->hostProfile.interDispatchGapMaxNs, uint64_t(gap));
        ++q->hostProfile.interDispatchGapCount;
      }
    }
  }
  OpenCLDispatchEnd dispatchEnd{q};
  auto executeStart = q->hostProfiling ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
  if (q->failed)
    return deviceAssertionStatus;
  if (config->mode != M_DRIVER_QUEUE_EXECUTE_MODE_GPU)
    return M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED;
  auto &cfg = config->config.gpu;
  if ((cfg.shared_mem_bytes && !fn->dynamicShared) || cfg.num_attributes)
    return M_DRIVER_STATUS_UNSUPPORTED;
  if (fn->dynamicShared && !cfg.shared_mem_bytes)
    return M_DRIVER_STATUS_INVALID_ARG;
  const size_t local[] = {cfg.block.x, cfg.block.y, cfg.block.z};
  const size_t groups[] = {cfg.grid.x, cfg.grid.y, cfg.grid.z};
  size_t global[3];
  for (int i = 0; i < 3; ++i) {
    if (!local[i] || !groups[i] || groups[i] > SIZE_MAX / local[i])
      return M_DRIVER_STATUS_INVALID_ARG;
    global[i] = groups[i] * local[i];
  }
  auto functionWaitStart =
      q->hostProfiling ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{};
  std::lock_guard<std::mutex> lock(fn->mutex);
  if (q->hostProfiling)
    q->hostProfile.functionLockNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - functionWaitStart)
            .count();
  auto &api = *q->context->device->api;
  const bool capture = q->recordQueue && !q->recordingDisabled &&
                       !q->profiling;
  M_driver_captured_dispatch *captured = nullptr;
  size_t capturedArgCount = 0;
  if (capture) {
    if (q->pendingDispatchCount == q->pendingDispatches.size())
      q->pendingDispatches.emplace_back();
    captured = &q->pendingDispatches[q->pendingDispatchCount];
    captured->kernel = fn->native;
    std::copy(std::begin(global), std::end(global), captured->global);
    std::copy(std::begin(local), std::end(local), captured->local);
  }
  auto setKernelArg = [&](cl_uint index, size_t size,
                          const void *value) -> cl_int {
    if (q->hostProfiling)
      ++q->hostProfile.setArgCalls;
    auto callStart = q->hostProfiling ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    auto status = q->recordingReady
                      ? CL_SUCCESS
                      : api.clSetKernelArg(fn->native, index, size, value);
    if (q->hostProfiling)
      q->hostProfile.setArgNs +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - callStart)
              .count();
    if (status || !capture)
      return status;
    if (capturedArgCount == captured->args.size())
      captured->args.emplace_back();
    auto &arg = captured->args[capturedArgCount++];
    arg.index = index;
    if (value) {
      auto *begin = static_cast<const unsigned char *>(value);
      arg.value.assign(begin, begin + size);
    } else {
      arg.value.assign(size, 0);
    }
    return CL_SUCCESS;
  };
  if (fn->subgroupSize) {
    if (!api.subgroupInfo)
      return M_DRIVER_STATUS_UNSUPPORTED;
    size_t nativeSize = 0, subgroupCount = 0;
    // CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE (OpenCL 2.1 / cl_khr_subgroups).
    auto status = api.subgroupInfo(fn->native, q->context->device->native,
                                   0x2033, sizeof(local), local,
                                   sizeof(nativeSize), &nativeSize, nullptr);
    if (status)
      return result(status);
    // Also check the count: a maximum of 64 alone does not prove that every
    // subgroup has 64 participating lanes.
    status = api.subgroupInfo(fn->native, q->context->device->native, 0x2034,
                              sizeof(local), local, sizeof(subgroupCount),
                              &subgroupCount, nullptr);
    if (status)
      return result(status);
    if (local[0] > SIZE_MAX / local[1] ||
        local[0] * local[1] > SIZE_MAX / local[2])
      return M_DRIVER_STATUS_INVALID_ARG;
    if (nativeSize != fn->subgroupSize ||
        (local[0] * local[1] * local[2]) % fn->subgroupSize ||
        subgroupCount != (local[0] * local[1] * local[2]) / fn->subgroupSize)
      return M_DRIVER_STATUS_UNSUPPORTED;
  }
  cl_uint expectedArgs = 0;
  auto metadataStart = q->hostProfiling ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
  auto error =
      api.clGetKernelInfo(fn->native, CL_KERNEL_NUM_ARGS, sizeof(expectedArgs),
                          &expectedArgs, nullptr);
  if (q->hostProfiling) {
    ++q->hostProfile.metadataCalls;
    q->hostProfile.metadataNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - metadataStart)
            .count();
  }
  if (error)
    return result(error);
  if (count != (fn->strictABI ? fn->arguments.size() : expectedArgs))
    return M_DRIVER_STATUS_INVALID_ARG;
  // Retain validated native buffers through enqueue, then let the command own
  // their lifetime. Handle bookkeeping need not stay locked during enqueue.
  MemoryReferences references{api};
  std::unique_lock<std::mutex> memoryLock(q->context->memoryMutex);
  std::vector<M::OpenCL::ArgumentABI> leaves;
  std::vector<const void *> leafValues;
  std::vector<uint64_t> leafSizes;
  for (uint32_t i = 0; i < count; ++i) {
    if (!sizes[i] || sizes[i] > SIZE_MAX || !args[i])
      return M_DRIVER_STATUS_INVALID_ARG;
    if (fn->strictABI && sizes[i] != fn->arguments[i].size)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (fn->strictABI &&
        fn->arguments[i].kind == M::OpenCL::ArgumentABI::Aggregate) {
      auto &root = fn->arguments[i];
      for (auto field : root.fields) {
        uint64_t offset = field.kind == M::OpenCL::ArgumentABI::BufferView
                              ? root.size / 4 + 3 * uint64_t(field.offset)
                              : field.offset;
        if (offset > sizes[i] || field.size > sizes[i] - offset)
          return M_DRIVER_STATUS_INVALID_ARG;
        leaves.push_back({field.kind, field.size});
        leafValues.push_back(static_cast<const char *>(args[i]) + offset);
        leafSizes.push_back(field.size);
      }
    } else {
      leaves.push_back(
          fn->strictABI ? fn->arguments[i]
                        : M::OpenCL::ArgumentABI{M::OpenCL::ArgumentABI::Scalar,
                                                 uint32_t(sizes[i])});
      leafValues.push_back(args[i]);
      leafSizes.push_back(sizes[i]);
    }
  }
  cl_uint nativeIndex = 0;
  const bool openclDebug = std::getenv("MODULAR_OPENCL_DEBUG") != nullptr;
  for (size_t i = 0; i < leaves.size(); ++i) {
    cl_int error;
    if (fn->strictABI && leaves[i].kind == M::OpenCL::ArgumentABI::BufferView) {
      static_assert(sizeof(M_driver_memory_view) == 24);
      M_driver_memory_view bufferView;
      std::memcpy(&bufferView, leafValues[i], sizeof(bufferView));
      if (openclDebug) {
        std::fprintf(stderr,
                     "OPENCL_DEBUG leaf=%zu memory=%p offset=%llu "
                     "size=%llu registered=%zu:",
                     i, static_cast<void *>(bufferView.memory),
                     (unsigned long long)bufferView.offset,
                     (unsigned long long)bufferView.size,
                     q->context->memories.size());
        for (auto *mem : q->context->memories)
          std::fprintf(stderr, " %p(%llu)", static_cast<void *>(mem),
                       (unsigned long long)mem->size);
        std::fprintf(stderr, "\n");
      }
      if (!bufferView.memory) {
        // OpenCL accepts a null cl_mem kernel argument. Only the canonical
        // empty view represents null; nonzero offsets/extents are malformed.
        if (bufferView.offset || bufferView.size)
          return M_DRIVER_STATUS_BAD_MEMORY;
        cl_mem nullMemory = nullptr;
        error = setKernelArg(nativeIndex++, sizeof(cl_mem), &nullMemory);
        if (!error)
          error = setKernelArg(nativeIndex++, sizeof(cl_ulong),
                               &bufferView.offset);
      } else {
        if (!q->context->memories.count(bufferView.memory) ||
            !view(q->context, &bufferView)) {
          if (openclDebug)
            std::fprintf(stderr, "OPENCL_DEBUG REJECT fn=%s\n",
                         fn->name.c_str());
          return M_DRIVER_STATUS_BAD_MEMORY;
        }
        auto retainStart = q->hostProfiling ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        error = references.retain(bufferView.memory->native);
        if (q->hostProfiling) {
          ++q->hostProfile.retainCalls;
          q->hostProfile.retainNs +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - retainStart)
                  .count();
        }
        if (error)
          return result(error);
        error = setKernelArg(nativeIndex++, sizeof(cl_mem),
                             &bufferView.memory->native);
        if (!error)
          error = setKernelArg(nativeIndex++, sizeof(cl_ulong),
                               &bufferView.offset);
      }
    } else
      error = setKernelArg(nativeIndex++, leafSizes[i], leafValues[i]);
    if (error)
      return result(error);
  }
  memoryLock.unlock();
  if (fn->dynamicShared) {
    error = setKernelArg(nativeIndex++, cfg.shared_mem_bytes, nullptr);
    if (error)
      return result(error);
    cl_ulong used = 0, limit = 0;
    error = api.clGetKernelWorkGroupInfo(fn->native, q->context->device->native,
                                         CL_KERNEL_LOCAL_MEM_SIZE, sizeof(used),
                                         &used, nullptr);
    if (error)
      return result(error);
    error = api.clGetDeviceInfo(q->context->device->native,
                                CL_DEVICE_LOCAL_MEM_SIZE, sizeof(limit), &limit,
                                nullptr);
    if (error)
      return result(error);
    if (used > limit)
      return M_DRIVER_STATUS_INVALID_ARG;
  }
  if (fn->deviceErrors) {
    if (!q->deviceError) {
      cl_uint zero = 0;
      q->deviceError = api.clCreateBuffer(
          q->context->native, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          sizeof(zero), &zero, &error);
      if (error)
        return result(error);
    }
    error = setKernelArg(nativeIndex++, sizeof(q->deviceError), &q->deviceError);
    if (error)
      return result(error);
  }
  cl_event event = nullptr;
  if (capture)
    captured->args.resize(capturedArgCount);
  auto enqueueStart = q->hostProfiling ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
  if (!q->recordingReady)
    error = api.clEnqueueNDRangeKernel(q->native, fn->native, 3, nullptr,
                                       global, local, 0, nullptr,
                                       q->profiling ? &event : nullptr);
  if (q->hostProfiling) {
    if (!q->recordingReady)
      ++q->hostProfile.enqueueCalls;
    q->hostProfile.enqueueNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - enqueueStart)
            .count();
  }
  if (!error && q->flushAfterEnqueue && !q->recordingReady) {
    auto flushStart = q->hostProfiling ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
    error = api.clFlush(q->native);
    if (q->hostProfiling) {
      ++q->hostProfile.flushCalls;
      q->hostProfile.flushNs +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - flushStart)
              .count();
    }
  }
  if (q->hostProfiling) {
    q->hostProfile.executeNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - executeStart)
            .count();
  }
  if (!error && capture)
    ++q->pendingDispatchCount;
  if (!error) {
    ++q->dispatchCount;
    if (q->hostProfiling)
      ++q->hostProfile.launches;
  }
  if (error == CL_SUCCESS && q->profiling) {
    std::lock_guard<std::mutex> lock(q->profileMutex);
    if (q->samples.size() < 100000)
      q->samples.push_back({fn->name, event});
    else {
      ++q->droppedSamples;
      api.clReleaseEvent(event);
    }
  }
  return result(error);
}
M_driver_status_t M_driver_event_create(M_driver_context_t ctx,
                                        M_driver_event_flags_t flags,
                                        M_driver_event_t *output) {
  if (!ctx || !output || (flags & ~M_DRIVER_EVENT_FLAGS_ALL))
    return M_DRIVER_STATUS_INVALID_ARG;
  *output = new M_driver_event{ctx};
  return M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_event_destroy(M_driver_context_t ctx,
                                         M_driver_event_t event) {
  if (!event || event->context != ctx)
    return M_DRIVER_STATUS_BAD_EVENT;
  auto error = event->native ? ctx->device->api->clReleaseEvent(event->native)
                             : CL_SUCCESS;
  delete event;
  return result(error);
}
M_driver_status_t M_driver_queue_record_event(M_driver_queue_t q,
                                              M_driver_event_t event) {
  if (!q || !event || event->context != q->context)
    return M_DRIVER_STATUS_BAD_EVENT;
  std::lock_guard<std::mutex> guard(q->errorMutex);
  auto &api = *q->context->device->api;
  auto error = updateQueueError(q);
  if (error)
    return result(error);
  if (q->failed)
    return deviceAssertionStatus;
  error = flushRecordedDispatches(q);
  if (error)
    return result(error);
  auto readback = std::shared_ptr<DeviceErrorReadback>();
  if (q->deviceError || q->lastError || !q->dependencies.empty()) {
    readback = std::make_shared<DeviceErrorReadback>();
    readback->dependencies = q->dependencies;
    if (q->lastError)
      readback->dependencies.push_back(q->lastError);
  }
  cl_event next = nullptr;
  if (q->deviceError)
    error = api.clEnqueueReadBuffer(q->native, q->deviceError, CL_FALSE, 0,
                                    sizeof(readback->code), &readback->code, 0,
                                    nullptr, &next);
  else
    error = api.clEnqueueMarkerWithWaitList(q->native, 0, nullptr, &next);
  if (error)
    return result(error);
  if (readback) {
    auto *keepalive = new std::shared_ptr<DeviceErrorReadback>(readback);
    error = api.clSetEventCallback(next, CL_COMPLETE, releaseErrorReadback,
                                   keepalive);
    if (error) {
      api.clWaitForEvents(1, &next);
      delete keepalive;
      api.clReleaseEvent(next);
      return result(error);
    }
    error = api.clRetainEvent(next);
    if (error) {
      api.clReleaseEvent(next);
      return result(error);
    }
  }
  if (q->lastErrorEvent)
    api.clReleaseEvent(q->lastErrorEvent);
  q->lastErrorEvent = readback ? next : nullptr;
  q->lastError = readback;
  q->dependencies.clear();
  if (event->native)
    api.clReleaseEvent(event->native);
  event->native = next;
  event->failure = readback;
  return result(api.clFlush(q->native));
}
M_driver_status_t M_driver_event_synchronize(M_driver_context_t ctx,
                                             M_driver_event_t event) {
  if (!event || event->context != ctx || !event->native)
    return M_DRIVER_STATUS_BAD_EVENT;
  auto error = ctx->device->api->clWaitForEvents(1, &event->native);
  if (error)
    return result(error);
  return readbackFailed(event->failure) ? deviceAssertionStatus
                                        : M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_is_event_ready(M_driver_context_t ctx,
                                          M_driver_event_t event, bool *ready) {
  if (!event || event->context != ctx || !event->native || !ready)
    return M_DRIVER_STATUS_BAD_EVENT;
  cl_int state;
  auto error = ctx->device->api->clGetEventInfo(
      event->native, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(state), &state,
      nullptr);
  if (error)
    return result(error);
  if (state < 0)
    return result(state);
  *ready = state == CL_COMPLETE;
  return *ready && readbackFailed(event->failure) ? deviceAssertionStatus
                                                  : M_DRIVER_STATUS_SUCCESS;
}
M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t q,
                                                 M_driver_event_t *events,
                                                 uint32_t count) {
  if (!q || (count && !events))
    return M_DRIVER_STATUS_INVALID_ARG;
  std::lock_guard<std::mutex> guard(q->errorMutex);
  std::vector<cl_event> native;
  std::vector<std::shared_ptr<DeviceErrorReadback>> dependencies;
  for (uint32_t i = 0; i < count; ++i) {
    if (!events[i] || events[i]->context != q->context || !events[i]->native)
      return M_DRIVER_STATUS_BAD_EVENT;
    native.push_back(events[i]->native);
    if (events[i]->failure)
      dependencies.push_back(events[i]->failure);
  }
  auto error = flushRecordedDispatches(q);
  if (error)
    return result(error);
  error = q->context->device->api->clEnqueueBarrierWithWaitList(
      q->native, count, count ? native.data() : nullptr, nullptr);
  if (!error)
    q->dependencies.insert(q->dependencies.end(), dependencies.begin(),
                           dependencies.end());
  return result(error);
}
