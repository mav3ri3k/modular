// Wide rows exercise split-K scratch transport and subgroup reductions.
// With ones: softmax = 1/65536 everywhere; argmax = 0 (lowest-index tie).
mo.graph @wide_softmax(%arg0: !mo.tensor<[1, 65536], f32>) -> !mo.tensor<[1, 65536], f32> attributes {_kernel_library_paths = [], argument_names = ["input0"], result_names = ["wide_softmax"]} {
  %0 = mo.chain.create()
  %result, %outChain = rmo.mo.transfer[%0] %arg0 : !mo.tensor<[1, 65536], f32> to <"gpu", 0>
  %1 = rmo.mo.reduce.softmax(%result) [axis = 1] : (!mo.tensor<[1, 65536], f32, gpu:0>) -> !mo.tensor<[1, 65536], f32, gpu:0>
  %result_0, %outChain_1 = rmo.mo.transfer[%0] %1 : !mo.tensor<[1, 65536], f32, gpu:0> to <"cpu", 0>
  mo.output %result_0 : !mo.tensor<[1, 65536], f32>
}
mo.graph @wide_argmax(%arg0: !mo.tensor<[1, 65536], f32>) -> !mo.tensor<[1, 1], si64> attributes {_kernel_library_paths = [], argument_names = ["input0"], result_names = ["wide_argmax"]} {
  %0 = mo.chain.create()
  %result, %outChain = rmo.mo.transfer[%0] %arg0 : !mo.tensor<[1, 65536], f32> to <"gpu", 0>
  %1 = rmo.mo.reduce.arg_max(%result) [axis = 1] : (!mo.tensor<[1, 65536], f32, gpu:0>) -> !mo.tensor<[1, 1], si64, gpu:0>
  %result_0, %outChain_1 = rmo.mo.transfer[%0] %1 : !mo.tensor<[1, 1], si64, gpu:0> to <"cpu", 0>
  mo.output %result_0 : !mo.tensor<[1, 1], si64>
}
