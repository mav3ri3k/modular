// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt Standalone host for Mojo-generated OpenCL. No
// handwritten GPU arithmetic.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#ifdef PROJECTION_KGSL_COUNTERS
#include "counters.h"
#endif

static void check(cl_int e) {
  if (e != CL_SUCCESS) {
    fprintf(stderr, "OpenCL error %d\n", e);
    exit(2);
  }
}
static double now() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static uint32_t randomWord(uint32_t &s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}
static uint16_t halfBits(float f) {
  _Float16 h = f;
  uint16_t b;
  memcpy(&b, &h, 2);
  return b;
}
static float halfFloat(uint16_t b) {
  _Float16 h;
  memcpy(&h, &b, 2);
  return float(h);
}
static uint32_t planes(uint32_t w) {
  uint32_t p = 0;
  for (int j = 0; j < 8; ++j) {
    p |= ((w >> (4 * j)) & 3u) << (2 * j);
    p |= ((w >> (4 * j + 2)) & 3u) << (16 + 2 * j);
  }
  return p;
}
static uint32_t unplanes(uint32_t p) {
  uint32_t w = 0;
  for (int j = 0; j < 8; ++j) {
    w |= ((p >> (2 * j)) & 3u) << (4 * j);
    w |= ((p >> (16 + 2 * j)) & 3u) << (4 * j + 2);
  }
  return w;
}
static size_t indexFor(const std::string &layout, int g, int word, int i,
                       int width, int groups, int outputBlock) {
  if (layout == "baseline")
    return size_t(g * 8 + word) * width + i;
  size_t tile = layout == "group_major"
                    ? size_t(g / 8) * (width / outputBlock) + i / outputBlock
                    : size_t(i / outputBlock) * (groups / 8) + g / 8;
  int inner =
      layout.find("group_word") == 0 ? (g % 8) * 8 + word : word * 8 + g % 8;
  return (tile * 64 + inner) * outputBlock + i % outputBlock;
}
struct Program {
  cl_program p;
  cl_kernel k;
};
static Program build(cl_context ctx, cl_device_id dev, const char *path) {
  std::ifstream f(path);
  std::string src((std::istreambuf_iterator<char>(f)), {});
  if (src.empty()) {
    fprintf(stderr, "Empty source: %s\n", path);
    exit(2);
  }
  const char *c = src.c_str();
  size_t n = src.size();
  cl_int e;
  cl_program p = clCreateProgramWithSource(ctx, 1, &c, &n, &e);
  check(e);
  e = clBuildProgram(p, 1, &dev, "-cl-std=CL3.0", nullptr, nullptr);
  if (e) {
    size_t len = 0;
    clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
    std::vector<char> log(len + 1);
    clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, len, log.data(),
                          nullptr);
    fprintf(stderr, "%s\n", log.data());
    check(e);
  }
  cl_uint count = 0;
  check(clCreateKernelsInProgram(p, 0, nullptr, &count));
  if (count != 1) {
    fprintf(stderr, "Expected one kernel, got %u\n", count);
    exit(2);
  }
  cl_kernel k;
  check(clCreateKernelsInProgram(p, 1, &k, nullptr));
  return {p, k};
}
static cl_mem buffer(cl_context ctx, size_t bytes, void *data = nullptr) {
  cl_int e;
  auto b =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE | (data ? CL_MEM_COPY_HOST_PTR : 0),
                     bytes, data, &e);
  check(e);
  return b;
}
// Generated ABI v3 pointer arguments are (buffer, byte offset).
static void args(cl_kernel k, cl_mem x, cl_mem w, cl_mem p, cl_mem y,
                 cl_ulong offset, int width, int reduction,
                 cl_ulong scaleOffset = 0) {
  cl_ulong zero = 0;
  check(clSetKernelArg(k, 0, sizeof(x), &x));
  check(clSetKernelArg(k, 1, sizeof(zero), &zero));
  check(clSetKernelArg(k, 2, sizeof(w), &w));
  check(clSetKernelArg(k, 3, sizeof(offset), &offset));
  check(clSetKernelArg(k, 4, sizeof(p), &p));
  check(clSetKernelArg(k, 5, sizeof(scaleOffset), &scaleOffset));
  check(clSetKernelArg(k, 6, sizeof(y), &y));
  check(clSetKernelArg(k, 7, sizeof(zero), &zero));
  check(clSetKernelArg(k, 8, sizeof(width), &width));
  check(clSetKernelArg(k, 9, sizeof(reduction), &reduction));
}
static cl_event enqueue(cl_command_queue q, cl_kernel k, int width, int groups,
                        bool reference = false) {
  int tile = !reference && getenv("PROJECTION_TILE")
                 ? atoi(getenv("PROJECTION_TILE"))
                 : 8;
  int joined = !reference && getenv("PROJECTION_GROUPS")
                   ? atoi(getenv("PROJECTION_GROUPS"))
                   : 8;
  size_t workgroup = !reference && getenv("PROJECTION_WORKGROUP")
                         ? atoi(getenv("PROJECTION_WORKGROUP"))
                         : 64;
  int outputs = tile * (64 / joined);
  size_t gx = size_t((width + outputs - 1) / outputs) * 64;
  gx = (gx + workgroup - 1) / workgroup * workgroup;
  size_t global[2] = {gx, size_t(groups / joined)}, local[2] = {workgroup, 1};
  cl_event ev;
  check(
      clEnqueueNDRangeKernel(q, k, 2, nullptr, global, local, 0, nullptr, &ev));
  return ev;
}
static double stamp(cl_event e, cl_profiling_info what) {
  cl_ulong v;
  check(clGetEventProfilingInfo(e, what, sizeof(v), &v, nullptr));
  return v * 1e-6;
}
int main(int argc, char **argv) {
  if (argc != 10) {
    fprintf(stderr, "bench reference.cl candidate.cl layout mode width "
                    "reduction banks iterations seed\n");
    return 2;
  }
  setbuf(stdout, nullptr);
  std::string layout = argv[3], mode = argv[4];
  int width = atoi(argv[5]), reduction = atoi(argv[6]), banks = atoi(argv[7]),
      iterations = atoi(argv[8]);
  uint32_t seed = strtoul(argv[9], nullptr, 10);
  int groups = reduction / 128;
  int outputBlock = getenv("PROJECTION_PACK_OUTPUT")
                        ? atoi(getenv("PROJECTION_PACK_OUTPUT"))
                        : 64;
  if (width <= 0 || width % 64 || reduction <= 0 || reduction % 1024 ||
      banks < 1 || iterations < banks)
    return 2;
  cl_platform_id platform;
  cl_device_id dev;
  cl_uint n;
  check(clGetPlatformIDs(1, &platform, &n));
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, &n));
  cl_int e;
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
  check(e);
  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
                                 0};
  cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &e);
  check(e);
  double t = now();
  printf("{\"stage\":\"reference_build_begin\"}\n");
  Program ref = build(ctx, dev, argv[1]);
  double referenceBuild = now() - t;
  t = now();
  printf("{\"stage\":\"candidate_build_begin\"}\n");
  Program candidate = build(ctx, dev, argv[2]);
  double candidateBuild = now() - t;
  printf("{\"stage\":\"built\",\"reference_ms\":%.6f,\"candidate_ms\":%.6f}\n",
         referenceBuild, candidateBuild);
  int joined =
      getenv("PROJECTION_GROUPS") ? atoi(getenv("PROJECTION_GROUPS")) : 8;
  if ((joined != 4 && joined != 8 && joined != 16) || groups % joined ||
      (mode != "full" && joined != 8))
    return 2;
  size_t words = size_t(groups) * 8 * width, bytes = words * 4,
         outCount = size_t(groups / joined) * width;
  size_t referenceCount = size_t(groups / 8) * width;
  bool pipeline = getenv("PROJECTION_REDUCE_SOURCE") != nullptr;
  Program reducer{nullptr, nullptr};
  if (pipeline)
    reducer = build(ctx, dev, getenv("PROJECTION_REDUCE_SOURCE"));
  std::vector<uint32_t> x(groups * 64), canonical(words), packed(words);
  std::vector<uint16_t> scales(size_t(groups) * width);
  std::vector<float> actual(outCount), expected(referenceCount),
      reducedValues(width);
  bool tiledScales = getenv("PROJECTION_TILED_SCALES") != nullptr;
  bool randomInputs = getenv("PROJECTION_RANDOM_INPUTS") != nullptr;
  for (auto &v : x) {
    uint16_t lo = halfBits(float(int(randomWord(seed) % 33) - 16) / 32);
    uint16_t hi = halfBits(float(int(randomWord(seed) % 33) - 16) / 32);
    if (randomInputs) {
      lo = halfBits(float(int(randomWord(seed) % 20001) - 10000) / 20003.f);
      hi = halfBits(float(int(randomWord(seed) % 20001) - 10000) / 20003.f);
    }
    v = lo | (uint32_t(hi) << 16);
  }
  for (auto &v : scales)
    v = halfBits(float(1 + randomWord(seed) % 8) / 16);
  std::vector<uint16_t> packedScales(scales.size());
  for (int g = 0; g < groups; ++g)
    for (int i = 0; i < width; ++i) {
      size_t dst =
          tiledScales
              ? ((size_t(g / 8) * (width / 64) + i / 64) * 8 + g % 8) * 64 +
                    i % 64
              : size_t(g) * width + i;
      packedScales[dst] = scales[size_t(g) * width + i];
    }
  cl_mem canonicalParams = buffer(ctx, scales.size() * 2, scales.data());
  cl_ulong privateBytes = 0, localBytes = 0;
  size_t maxWorkgroup = 0;
  check(clGetKernelWorkGroupInfo(candidate.k, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
                                 sizeof(privateBytes), &privateBytes, nullptr));
  check(clGetKernelWorkGroupInfo(candidate.k, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                                 sizeof(localBytes), &localBytes, nullptr));
  check(clGetKernelWorkGroupInfo(candidate.k, dev, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(maxWorkgroup), &maxWorkgroup, nullptr));
  printf("{\"stage\":\"resources\",\"private_bytes\":%llu,\"local_bytes\":%llu,"
         "\"max_workgroup\":%zu}\n",
         (unsigned long long)privateBytes, (unsigned long long)localBytes,
         maxWorkgroup);
  cl_mem xb = buffer(ctx, x.size() * 4, x.data()),
         pb = buffer(ctx, scales.size() * 2 * banks);
  cl_mem wb = buffer(ctx, bytes * banks), rb = buffer(ctx, bytes),
         yb = buffer(ctx, outCount * 4), ry = buffer(ctx, referenceCount * 4);
  cl_mem reducedOutput = buffer(ctx, size_t(width) * 4);
  if (pipeline) {
    cl_ulong zero = 0;
    int parts = groups / joined;
    check(clSetKernelArg(reducer.k, 0, sizeof(yb), &yb));
    check(clSetKernelArg(reducer.k, 1, sizeof(zero), &zero));
    check(clSetKernelArg(reducer.k, 2, sizeof(reducedOutput), &reducedOutput));
    check(clSetKernelArg(reducer.k, 3, sizeof(zero), &zero));
    check(clSetKernelArg(reducer.k, 4, sizeof(width), &width));
    check(clSetKernelArg(reducer.k, 5, sizeof(parts), &parts));
  }
  auto reduceEnqueue = [&]() -> cl_event {
    cl_event ev;
    size_t local = 128, global = (size_t(width) + 127) / 128 * 128;
    check(clEnqueueNDRangeKernel(q, reducer.k, 1, nullptr, &global, &local, 0,
                                 nullptr, &ev));
    return ev;
  };
  // Validate every bank before timing; repacking must be a bijection.
  double firstEnqueue = 0, firstFinish = 0, firstDevice = 0;
  for (int bank = 0; bank < banks; ++bank) {
    for (auto &v : canonical)
      v = randomWord(seed);
    std::vector<uint8_t> seen(words, 0);
    for (int g = 0; g < groups; ++g)
      for (int word = 0; word < 8; ++word)
        for (int i = 0; i < width; ++i) {
          size_t src = size_t(g * 8 + word) * width + i,
                 dst = indexFor(layout, g, word, i, width, groups, outputBlock);
          if (dst >= words || seen[dst]++) {
            fprintf(stderr, "Nonbijective layout\n");
            return 3;
          }
          uint32_t v = canonical[src];
          packed[dst] =
              layout.find("planes") != std::string::npos ? planes(v) : v;
          if ((layout.find("planes") != std::string::npos
                   ? unplanes(packed[dst])
                   : packed[dst]) != v)
            return 3;
        }
    check(clEnqueueWriteBuffer(q, pb, CL_TRUE, bank * scales.size() * 2,
                               scales.size() * 2, packedScales.data(), 0,
                               nullptr, nullptr));
    check(clEnqueueWriteBuffer(q, wb, CL_TRUE, bank * bytes, bytes,
                               packed.data(), 0, nullptr, nullptr));
    check(clEnqueueWriteBuffer(q, rb, CL_TRUE, 0, bytes, canonical.data(), 0,
                               nullptr, nullptr));
    args(candidate.k, xb, wb, pb, yb, bank * bytes, width, reduction,
         bank * scales.size() * 2);
    t = now();
    if (bank == 0)
      printf("{\"stage\":\"first_enqueue_begin\"}\n");
    auto ev = enqueue(q, candidate.k, width, groups);
    double enq = now() - t;
    if (bank == 0)
      printf("{\"stage\":\"first_enqueue_end\",\"ms\":%.6f}\n", enq);
    cl_event reductionEvent = pipeline ? reduceEnqueue() : nullptr;
    t = now();
    check(clFinish(q));
    if (reductionEvent)
      check(clReleaseEvent(reductionEvent));
    double fin = now() - t;
    if (bank == 0) {
      firstEnqueue = enq;
      firstFinish = fin;
      firstDevice = stamp(ev, CL_PROFILING_COMMAND_END) -
                    stamp(ev, CL_PROFILING_COMMAND_START);
    }
    check(clReleaseEvent(ev));
    check(clEnqueueReadBuffer(q, yb, CL_TRUE, 0, outCount * 4, actual.data(), 0,
                              nullptr, nullptr));
    if (mode == "full" || (bank > 0 && mode == "unpack")) {
      args(ref.k, xb, rb, canonicalParams, ry, 0, width, reduction, 0);
      auto re = enqueue(q, ref.k, width, groups, true);
      check(clEnqueueReadBuffer(q, ry, CL_TRUE, 0, referenceCount * 4,
                                expected.data(), 0, nullptr, nullptr));
      check(clReleaseEvent(re));
    } else {
      std::fill(expected.begin(), expected.end(), 0.f);
      for (int g = 0; g < groups; ++g)
        for (int i = 0; i < width; ++i) {
          uint32_t hash = 0;
          for (int word = 0; word < 8; ++word) {
            uint32_t codes = canonical[size_t(g * 8 + word) * width + i];
            for (int pair = 0; pair < 8; ++pair)
              hash ^= x[g * 64 + word * 8 + pair];
            if (mode == "load")
              hash ^= layout.find("planes") != std::string::npos ? planes(codes)
                                                                 : codes;
            else
              for (int pair = 0; pair < 8; ++pair) {
                int lo = (codes >> (pair * 4)) & 3,
                    hi = (codes >> (pair * 4 + 2)) & 3;
                hash = (hash * 16777619u) ^ halfBits(float(lo - 1));
                hash = (hash * 16777619u) ^ halfBits(float(hi - 1));
              }
          }
          expected[size_t(g / 8) * width + i] +=
              float((hash ^ (hash >> 16)) & 1023) *
              halfFloat(scales[size_t(g) * width + i]);
        }
    }
    if (mode == "full" && bank == 0 && !randomInputs) {
      // Independent CPU check on 16 output columns. Dyadic test inputs keep
      // every FP16 accumulation exact, so the FP32 oracle is exact too.
      for (int gtile = 0; gtile < groups / joined; ++gtile)
        for (int sample = 0; sample < 16; ++sample) {
          int i = sample < 8 ? sample : width - 16 + sample;
          float total = 0;
          for (int g = gtile * joined; g < (gtile + 1) * joined; ++g) {
            float dot = 0;
            for (int word = 0; word < 8; ++word) {
              uint32_t codes = canonical[size_t(g * 8 + word) * width + i];
              for (int pair = 0; pair < 8; ++pair) {
                uint32_t q = x[g * 64 + word * 8 + pair];
                dot += halfFloat(uint16_t(q)) *
                       (int((codes >> (pair * 4)) & 3) - 1);
                dot += halfFloat(uint16_t(q >> 16)) *
                       (int((codes >> (pair * 4 + 2)) & 3) - 1);
              }
            }
            total += dot * halfFloat(scales[size_t(g) * width + i]);
          }
          if (actual[size_t(gtile) * width + i] != total) {
            fprintf(stderr, "CPU oracle mismatch i=%d got=%g expected=%g\n", i,
                    actual[size_t(gtile) * width + i], total);
            return 3;
          }
        }
    }
    if (joined == 8) {
      for (size_t i = 0; i < outCount; ++i)
        if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
          fprintf(stderr, "Mismatch bank=%d i=%zu got=%g expected=%g\n", bank,
                  i, actual[i], expected[i]);
          return 3;
        }
    }
    if (pipeline)
      check(clEnqueueReadBuffer(q, reducedOutput, CL_TRUE, 0, size_t(width) * 4,
                                reducedValues.data(), 0, nullptr, nullptr));
    for (int i = 0; i < width; ++i) {
      float a = 0, b = 0;
      for (int g = 0; g < groups / joined; ++g)
        a += actual[size_t(g) * width + i];
      for (int g = 0; g < groups / 8; ++g)
        b += expected[size_t(g) * width + i];
      float tolerance = randomInputs ? 1e-5f + 1e-5f * std::abs(b) : 0;
      if (!std::isfinite(a) || std::abs(a - b) > tolerance ||
          (pipeline && (!std::isfinite(reducedValues[i]) ||
                        std::abs(reducedValues[i] - b) > tolerance))) {
        fprintf(stderr,
                "Final-output mismatch bank=%d i=%d got=%g expected=%g\n", bank,
                i, a, b);
        return 3;
      }
    }
  }
  printf(
      "{\"stage\":\"validated\",\"banks\":%d,\"outputs\":%zu,\"first_enqueue_"
      "ms\":%.6f,\"first_finish_ms\":%.6f,\"first_device_ms\":%.6f}\n",
      banks, outCount, firstEnqueue, firstFinish, firstDevice);
  std::vector<cl_event> events, reductionEvents;
  std::vector<double> enqueueTimes;
  // Extra warmup excludes setup/validation and touches all resident banks.
  for (int j = 0; j < std::max(banks, getenv("PROJECTION_WARMUP")
                                          ? atoi(getenv("PROJECTION_WARMUP"))
                                          : 256);
       ++j) {
    args(candidate.k, xb, wb, pb, yb, (j % banks) * bytes, width, reduction,
         (j % banks) * scales.size() * 2);
    auto ev = enqueue(q, candidate.k, width, groups);
    check(clReleaseEvent(ev));
    if (pipeline)
      check(clReleaseEvent(reduceEnqueue()));
  }
  check(clFinish(q));
#ifdef PROJECTION_KGSL_COUNTERS
  ProjectionCounters *counters = nullptr;
  if (getenv("PROJECTION_COUNTERS")) {
    counters = new ProjectionCounters;
    counters->snapshot(true);
  }
#endif
  t = now();
  for (int j = 0; j < iterations; ++j) {
    args(candidate.k, xb, wb, pb, yb, (j % banks) * bytes, width, reduction,
         (j % banks) * scales.size() * 2);
    double begin = now();
    events.push_back(enqueue(q, candidate.k, width, groups));
    if (pipeline)
      reductionEvents.push_back(reduceEnqueue());
    enqueueTimes.push_back(now() - begin);
  }
  check(clFinish(q));
  double wall = now() - t;
#ifdef PROJECTION_KGSL_COUNTERS
  if (counters) {
    int tile = getenv("PROJECTION_TILE") ? atoi(getenv("PROJECTION_TILE")) : 8;
    int wg = getenv("PROJECTION_WORKGROUP")
                 ? atoi(getenv("PROJECTION_WORKGROUP"))
                 : 64;
    size_t gx =
        size_t((width + tile * (64 / joined) - 1) / (tile * (64 / joined))) *
        64;
    gx = (gx + wg - 1) / wg * wg;
    unsigned long long expected =
        iterations * (gx * (groups / joined) +
                      (pipeline ? (size_t(width) + 127) / 128 * 128 : 0));
    counters->snapshot(false, expected);
    delete counters;
  }
#endif

  std::vector<double> durations, submitWait, queueWait;
  double start = stamp(events.front(), CL_PROFILING_COMMAND_START),
         end = stamp(pipeline ? reductionEvents.back() : events.back(),
                     CL_PROFILING_COMMAND_END);
  double projectionSum = 0, reductionSum = 0;
  size_t eventIndex = 0;
  for (auto ev : events) {
    double queued = stamp(ev, CL_PROFILING_COMMAND_QUEUED),
           submitted = stamp(ev, CL_PROFILING_COMMAND_SUBMIT),
           s = stamp(ev, CL_PROFILING_COMMAND_START),
           e = stamp(ev, CL_PROFILING_COMMAND_END);
    double rd = 0;
    if (pipeline) {
      auto re = reductionEvents[eventIndex];
      rd = stamp(re, CL_PROFILING_COMMAND_END) -
           stamp(re, CL_PROFILING_COMMAND_START);
      check(clReleaseEvent(re));
    }
    ++eventIndex;
    projectionSum += e - s;
    reductionSum += rd;
    durations.push_back(e - s + rd);
    submitWait.push_back(s - submitted);
    queueWait.push_back(submitted - queued);
    check(clReleaseEvent(ev));
  }
  double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
  std::sort(durations.begin(), durations.end());
  double enqSum =
      std::accumulate(enqueueTimes.begin(), enqueueTimes.end(), 0.0);
  printf("{\"stage\":\"result\",\"layout\":\"%s\",\"mode\":\"%s\",\"width\":%d,"
         "\"reduction\":%d,\"banks\":%d,\"iterations\":%d,\"weight_bytes\":%zu,"
         "\"resident_weight_bytes\":%zu,\"resident_scale_bytes\":%zu,\"device_"
         "mean_ms\":%.6f,\"device_median_ms\":%.6f,\"device_p95_ms\":%.6f,"
         "\"device_sum_ms\":%.6f,\"device_span_ms\":%.6f,\"host_wall_ms\":%.6f,"
         "\"enqueue_mean_ms\":%.6f,\"enqueue_max_ms\":%.6f,\"queued_to_submit_"
         "mean_ms\":%.6f,\"submit_to_start_mean_ms\":%.6f}\n",
         layout.c_str(), mode.c_str(), width, reduction, banks, iterations,
         bytes, bytes * banks, scales.size() * 2 * banks, sum / iterations,
         durations[iterations / 2],
         durations[std::min(iterations - 1, int(iterations * .95))], sum,
         end - start, wall, enqSum / iterations,
         *std::max_element(enqueueTimes.begin(), enqueueTimes.end()),
         std::accumulate(queueWait.begin(), queueWait.end(), 0.0) / iterations,
         std::accumulate(submitWait.begin(), submitWait.end(), 0.0) /
             iterations);
  printf("{\"stage\":\"components\",\"projection_mean_ms\":%.6f,\"reduction_"
         "mean_ms\":%.6f,\"pipeline\":%s}\n",
         projectionSum / iterations, reductionSum / iterations,
         pipeline ? "true" : "false");
  for (auto b : {xb, pb, canonicalParams, wb, rb, yb, ry, reducedOutput})
    check(clReleaseMemObject(b));
  for (auto p : {ref, candidate}) {
    check(clReleaseKernel(p.k));
    check(clReleaseProgram(p.p));
  }
  if (pipeline) {
    check(clReleaseKernel(reducer.k));
    check(clReleaseProgram(reducer.p));
  }
  check(clReleaseCommandQueue(q));
  check(clReleaseContext(ctx));
}
