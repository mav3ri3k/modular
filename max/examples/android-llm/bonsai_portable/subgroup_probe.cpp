// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <dlfcn.h>
int main() {
  void *lib =
      dlopen("/vendor/lib64/egl/mt6897/libGLES_mali.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    puts(dlerror());
    return 1;
  }
#define LOAD(n)                                                                \
  auto n = reinterpret_cast<decltype(&::n)>(dlsym(lib, #n));                   \
  if (!n) {                                                                    \
    printf("missing %s\n", #n);                                                \
    return 1;                                                                  \
  }
  LOAD(clGetPlatformIDs);
  LOAD(clGetDeviceIDs);
  LOAD(clCreateContext);
  LOAD(clCreateProgramWithSource);
  LOAD(clBuildProgram);
  LOAD(clGetProgramBuildInfo);
  LOAD(clCreateKernel);
  LOAD(clGetKernelSubGroupInfo);
  cl_platform_id p;
  cl_device_id d;
  cl_int e = clGetPlatformIDs(1, &p, nullptr);
  if (e)
    return printf("platform %d\n", e), 1;
  e = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, nullptr);
  if (e)
    return printf("device %d\n", e), 1;
  cl_context c = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &e);
  if (e)
    return printf("context %d\n", e), 1;
  const char *src =
      "__kernel void k(__global int *x) { x[get_global_id(0)] += 1; }";
  cl_program prog = clCreateProgramWithSource(c, 1, &src, nullptr, &e);
  if (e)
    return 1;
  e = clBuildProgram(prog, 1, &d, "-cl-std=CL3.0", nullptr, nullptr);
  if (e) {
    char log[8192]{};
    clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, sizeof(log), log,
                          nullptr);
    printf("build %d %s\n", e, log);
    return 1;
  }
  cl_kernel k = clCreateKernel(prog, "k", &e);
  size_t local = 64, width = 0, count = 0;
  e = clGetKernelSubGroupInfo(k, d, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                              sizeof(local), &local, sizeof(width), &width,
                              nullptr);
  printf("subgroup max: status=%d width=%zu\n", e, width);
  e = clGetKernelSubGroupInfo(k, d, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
                              sizeof(local), &local, sizeof(count), &count,
                              nullptr);
  printf("subgroup count: status=%d count=%zu\n", e, count);
}
