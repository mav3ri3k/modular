#include <stdint.h>
#include <stdio.h>

extern void mojo_android_matmul(float *a, float *b, float *output, int32_t m,
                                int32_t n, int32_t k);

int main(void) {
  float a[] = {1, 2, 3, 4, 5, 6};
  float b[] = {7, 8, 9, 10, 11, 12};
  float output[4] = {0};
  const float expected[] = {58, 64, 139, 154};
  mojo_android_matmul(a, b, output, 2, 2, 3);
  for (int i = 0; i < 4; ++i) {
    if (output[i] != expected[i]) {
      fprintf(stderr, "FAIL: output[%d]=%g, expected %g\n", i, output[i],
              expected[i]);
      return 1;
    }
  }
  puts("PASS: Mojo matrix multiplication called through the C ABI");
  return 0;
}
