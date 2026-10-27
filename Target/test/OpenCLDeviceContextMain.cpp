// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
// A native embedding entry point. All device operations belong to Mojo's
// ordinary DeviceContext; this executable has no HAL or kernel-launch code.
extern "C" int run_opencl_device_context();
extern "C" void KGEN_CompilerRT_DestroyGlobals();
int main() {
  int status = run_opencl_device_context();
  KGEN_CompilerRT_DestroyGlobals();
  return status;
}
