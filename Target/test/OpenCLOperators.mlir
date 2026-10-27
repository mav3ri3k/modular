// Compile with the normal Android/OpenCL graph pipeline, then run with mt.
// CPU inputs/outputs and explicit transfers exercise opaque graph storage.
// With ones inputs: matmul = 8, rmsnorm ~= 0.999995, softmax = 0.125, views = 1.
mo.graph @matmul(%arg0: !mo.tensor<[2, 8], f32>, %arg1: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[2, 8], f32> attributes {_kernel_library_paths = [], argument_names = ["input0", "input1"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %result, %outChain = rmo.mo.transfer[%0] %arg0 : !mo.tensor<[2, 8], f32> to <"gpu", 0>
  %result_0, %outChain_1 = rmo.mo.transfer[%0] %arg1 : !mo.tensor<[8, 8], f32> to <"gpu", 0>
  %1 = rmo.matmul(%result, %result_0) : (!mo.tensor<[2, 8], f32, gpu:0>, !mo.tensor<[8, 8], f32, gpu:0>) -> !mo.tensor<[2, 8], f32, gpu:0>
  %result_2, %outChain_3 = rmo.mo.transfer[%0] %1 : !mo.tensor<[2, 8], f32, gpu:0> to <"cpu", 0>
  mo.output %result_2 : !mo.tensor<[2, 8], f32>
}
mo.graph @rmsnorm(%arg0: !mo.tensor<[2, 8], f32>, %arg1: !mo.tensor<[8], f32>) -> !mo.tensor<[2, 8], f32> attributes {_kernel_library_paths = [], argument_names = ["input0", "input1"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %result, %outChain = rmo.mo.transfer[%0] %arg0 : !mo.tensor<[2, 8], f32> to <"gpu", 0>
  %result_0, %outChain_1 = rmo.mo.transfer[%0] %arg1 : !mo.tensor<[8], f32> to <"gpu", 0>
  %1 = mo.reduce.rms_norm(%result, %result_0) {epsilon = #M.dense_array<9.99999974E-6> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 8], f32, gpu:0>, !mo.tensor<[8], f32, gpu:0>) -> !mo.tensor<[2, 8], f32, gpu:0>
  %result_2, %outChain_3 = rmo.mo.transfer[%0] %1 : !mo.tensor<[2, 8], f32, gpu:0> to <"cpu", 0>
  mo.output %result_2 : !mo.tensor<[2, 8], f32>
}
mo.graph @softmax(%arg0: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 8], f32> attributes {_kernel_library_paths = [], argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %result, %outChain = rmo.mo.transfer[%0] %arg0 : !mo.tensor<[2, 8], f32> to <"gpu", 0>
  %1 = rmo.mo.reduce.softmax(%result) [axis = 1] : (!mo.tensor<[2, 8], f32, gpu:0>) -> !mo.tensor<[2, 8], f32, gpu:0>
  %result_0, %outChain_1 = rmo.mo.transfer[%0] %1 : !mo.tensor<[2, 8], f32, gpu:0> to <"cpu", 0>
  mo.output %result_0 : !mo.tensor<[2, 8], f32>
}
mo.graph @views(%arg0: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 8], f32> attributes {_kernel_library_paths = [], argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %result, %outChain = rmo.mo.transfer[%0] %arg0 : !mo.tensor<[2, 8], f32> to <"gpu", 0>
  %1 = rmo.slice(%result) [starts = #mosh<ape[0, 1]> : !mosh.ape, stops = #mosh<ape[2, 5]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape] : (!mo.tensor<[2, 8], f32, gpu:0>) -> !mo.tensor<[2, 4], f32, gpu:0>
  %2 = rmo.slice(%result) [starts = #mosh<ape[0, 1]> : !mosh.ape, stops = #mosh<ape[2, 5]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape] : (!mo.tensor<[2, 8], f32, gpu:0>) -> !mo.tensor<[2, 4], f32, gpu:0>
  %3 = rmo.concat(%1, %2) [axis = 1 : index] : (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>) -> !mo.tensor<[2, 8], f32, gpu:0>
  %result_0, %outChain_1 = rmo.mo.transfer[%0] %3 : !mo.tensor<[2, 8], f32, gpu:0> to <"cpu", 0>
  mo.output %result_0 : !mo.tensor<[2, 8], f32>
}