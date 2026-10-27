from std.memory import Pointer


@export("mojo_android_matmul")
def mojo_android_matmul(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
) abi("C"):
    for i in range(Int(m)):
        for j in range(Int(n)):
            var value: Float32 = 0
            for p in range(Int(k)):
                value += (
                    a[unsafe_offset=i * Int(k) + p]
                    * b[unsafe_offset=p * Int(n) + j]
                )
            output[unsafe_offset=i * Int(n) + j] = value
