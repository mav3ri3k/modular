// Cross-compile with the ordinary graph pipeline and execute the resulting MEF
// with Android mt. See Target/lib/OpenCL/README.md for the build/deploy commands.
// The explicit transfers exercise graph allocation and opaque buffer ownership.
// With --model-inputs=ones, both outputs contain six elements.
// CHECK: 'arithmetic' returned tensor<2x3xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK: tensor<2x3xf32> [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]
mo.graph @arithmetic(%a: !mo.tensor<[2, 3], f32>, %b: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) {
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[2, 3], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[2, 3], f32> to <"gpu", 0>
  %sum = mo.add(%a_gpu, %b_gpu) : !mo.tensor<[2, 3], f32, gpu:0>
  %product = mo.mul(%sum, %sum) : !mo.tensor<[2, 3], f32, gpu:0>
  %out_sum, %ch3 = mo.transfer[%ch2] %sum : !mo.tensor<[2, 3], f32, gpu:0> to <"cpu", 0>
  %out_product, %ch4 = mo.transfer[%ch3] %product : !mo.tensor<[2, 3], f32, gpu:0> to <"cpu", 0>
  mo.output %out_sum, %out_product : !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>
}
