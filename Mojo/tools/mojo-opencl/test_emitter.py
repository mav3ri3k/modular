"""Check that unsupported LLVM constructs fail closed before a bundle is emitted."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    emitter = str(Path(sys.argv[1]).resolve())
    preamble = 'target datalayout = "e-p:64:64"\ntarget triple = "aarch64-unknown-opencl"\n'
    cases = [
        (
            "nonvoid",
            "define i32 @bad() { ret i32 0 }",
            "kernel must return void",
        ),
        (
            "vector",
            "define void @bad(<5 x float> %x) { ret void }",
            "unsupported LLVM vector width",
        ),
        (
            "assembly",
            'define void @bad() { call void asm sideeffect "", ""()\n ret void }',
            "inline assembly",
        ),
        (
            "external",
            "declare void @unavailable()\ndefine void @bad() { call void @unavailable()\n ret void }",
            "unsupported external call",
        ),
        (
            "pointercast",
            "define void @bad(ptr %p, ptr %out) { %v = ptrtoint ptr %p to i64\n store i64 %v, ptr %out\n ret void }",
            "unsupported pointer cast",
        ),
        (
            "addressspace",
            "define void @bad(ptr addrspace(4) %p) { ret void }",
            "pointer address spaces",
        ),
        (
            "assertion-barrier",
            "declare void @llvm.trap()\ndeclare void @__mojo_opencl_barrier()\ndefine void @bad() { call void @__mojo_opencl_barrier()\n call void @llvm.trap()\n unreachable }",
            "device assertions with workgroup barriers",
        ),
        (
            "subgroup-barrier-signature",
            "declare void @__mojo_opencl_subgroup_barrier_w64(i32)\ndefine void @bad() { call void @__mojo_opencl_subgroup_barrier_w64(i32 0)\n ret void }",
            "invalid subgroup barrier signature",
        ),
        (
            "assertion-subgroup-barrier",
            "declare void @llvm.trap()\ndeclare void @__mojo_opencl_subgroup_barrier_w64()\ndefine void @bad() { call void @__mojo_opencl_subgroup_barrier_w64()\n call void @llvm.trap()\n unreachable }",
            "barriers require safe workgroup termination",
        ),
        (
            "packed-bool",
            "define void @bad(ptr %p, ptr %out) { %v = load <4 x i1>, ptr %p, align 1\n %w = zext <4 x i1> %v to <4 x i32>\n store <4 x i32> %w, ptr %out\n ret void }",
            "packed boolean vector memory",
        ),
    ]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for name, body, message in cases:
            source, output = work / f"{name}.ll", work / f"{name}.mcl"
            source.write_text(preamble + body)
            result = subprocess.run(
                [emitter, str(source), str(output), "--mark-exported"],
                text=True,
                capture_output=True,
            )
            if (
                result.returncode != 1
                or message not in result.stderr
                or output.exists()
            ):
                raise AssertionError(
                    f"{name}: status={result.returncode}\n{result.stderr}"
                )
            print(f"PASS: rejects {name}")

        # LLVM masked-memory tails can contain one-lane vectors. Legalize
        # these to scalars without inventing OpenCL C's nonexistent float1.
        source = work / "one-lane.ll"
        output, generated = work / "one-lane.mcl", work / "one-lane.cl"
        source.write_text(
            preamble + """
define void @one_lane(ptr %in, ptr %out, float %bias, i1 %flag) {
 %v = load <1 x float>, ptr %in, align 4
 %bias_vector = bitcast float %bias to <1 x float>
 %added = fadd <1 x float> %v, %bias_vector
 %sum = select i1 %flag, <1 x float> %added, <1 x float> zeroinitializer
 store <1 x float> %sum, ptr %out, align 4
 %broadcast = shufflevector <1 x float> %sum, <1 x float> poison, <4 x i32> zeroinitializer
 store <4 x float> %broadcast, ptr %out, align 4
 ret void
}
"""
        )
        subprocess.run(
            [emitter, str(source), str(output), str(generated), "--mark-exported"],
            check=True,
        )
        text = generated.read_text()
        assert "float1" not in text and "vload1" not in text and "vstore1" not in text
        assert "__kernel void mojo_kernel_" in text
        print("PASS: one-lane vector memory and arithmetic legalize to scalars")

        # OpenCL char/short comparison masks cannot be ANDed with an int
        # literal. Exercise both integer and floating narrow operands.
        source = work / "narrow-compare.ll"
        output, generated = (
            work / "narrow-compare.mcl",
            work / "narrow-compare.cl",
        )
        source.write_text(
            preamble
            + """
define void @narrow_compare(ptr %in, ptr %out) {
 %a = load <16 x i16>, ptr %in, align 2
 %b = icmp ugt <16 x i16> %a, zeroinitializer
 %c = zext <16 x i1> %b to <16 x i16>
 store <16 x i16> %c, ptr %out, align 2
 %f = load <8 x half>, ptr %in, align 2
 %p = fcmp ogt <8 x half> %f, zeroinitializer
 %q = zext <8 x i1> %p to <8 x i16>
 store <8 x i16> %q, ptr %out, align 2
 ret void
}
"""
        )
        subprocess.run(
            [
                emitter,
                str(source),
                str(output),
                str(generated),
                "--mark-exported",
            ],
            check=True,
        )
        if len(sys.argv) > 2:
            subprocess.run(
                [
                    sys.argv[2],
                    "--target=spir64-unknown-unknown",
                    "-x",
                    "cl",
                    "-cl-std=CL1.2",
                    "-Werror",
                    "-fsyntax-only",
                    str(generated),
                ],
                check=True,
            )
        print("PASS: narrow vector comparison masks compile as OpenCL C")

        # Shared LLVM optimizations can leave vectors wider than OpenCL C.
        # Use LLVM's scalarizer for these functions, preserving normal SIMD.
        source = work / "wide-vector.ll"
        output, generated = work / "wide-vector.mcl", work / "wide-vector.cl"
        source.write_text(preamble + """
define void @wide_vector(ptr %in, ptr %out) {
 %v = load <32 x float>, ptr %in, align 4
 %sum = fadd <32 x float> %v, %v
 store <32 x float> %sum, ptr %out, align 4
 ret void
}
define void @native_vector(ptr %in, ptr %out) {
 %v = load <8 x float>, ptr %in, align 4
 %sum = fadd <8 x float> %v, %v
 store <8 x float> %sum, ptr %out, align 4
 ret void
}
""")
        subprocess.run(
            [emitter, str(source), str(output), str(generated), "--mark-exported"],
            check=True,
        )
        text = generated.read_text()
        assert "float32" not in text and "float8" in text
        print("PASS: wide vector legalization preserves native SIMD functions")

        # GEP must retain address-space pointer provenance, including signed
        # displacements. Adreno otherwise loses shared-memory communication
        # between warps when local pointers round-trip through integers.
        source = work / "pointer-offset.ll"
        output, generated = work / "pointer-offset.mcl", work / "pointer-offset.cl"
        source.write_text(preamble + """
@shared = internal addrspace(3) global [16 x float] undef, align 4
declare void @__mojo_opencl_barrier()
define void @pointer_offset(ptr %out, i64 %index) {
 %slot = getelementptr [16 x float], ptr addrspace(3) @shared, i64 0, i64 %index
 store float 3.0, ptr addrspace(3) %slot, align 4
 call void @__mojo_opencl_barrier()
 %previous = getelementptr float, ptr addrspace(3) %slot, i64 -1
 %value = load float, ptr addrspace(3) %previous, align 4
 store float %value, ptr %out, align 4
 ret void
}
""")
        subprocess.run([emitter, str(source), str(output), str(generated), "--mark-exported"], check=True)
        text = generated.read_text()
        assert "((ulong)mojo_shared_" not in text
        assert "+((long)(" in text and "FFFFFFFFFFFFFFFCUL" in text
        if len(sys.argv) > 2:
            subprocess.run([sys.argv[2], "--target=spir64-unknown-unknown", "-x", "cl", "-cl-std=CL1.2", "-Werror", "-fsyntax-only", str(generated)], check=True)
        print("PASS: shared GEP preserves pointer provenance and signed offsets")

        source = work / "subgroup-barrier.ll"
        output, generated = work / "subgroup-barrier.mcl", work / "subgroup-barrier.cl"
        source.write_text(preamble + """
@shared = internal addrspace(3) global [256 x float] undef, align 4
declare void @__mojo_opencl_subgroup_barrier_w64() convergent noduplicate
define void @subgroup_barrier(ptr %out, i64 %index, i64 %partner) {
 %slot = getelementptr [256 x float], ptr addrspace(3) @shared, i64 0, i64 %index
 store float 3.0, ptr addrspace(3) %slot, align 4
 call void @__mojo_opencl_subgroup_barrier_w64()
 %other = getelementptr [256 x float], ptr addrspace(3) @shared, i64 0, i64 %partner
 %value = load float, ptr addrspace(3) %other, align 4
 store float %value, ptr %out, align 4
 ret void
}
""")
        subprocess.run([emitter, str(source), str(output), str(generated), "--mark-exported"], check=True)
        text = generated.read_text()
        assert "sub_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)" in text
        assert "\nbarrier(" not in text
        if len(sys.argv) > 2:
            subprocess.run([sys.argv[2], "--target=spir64-unknown-unknown", "-x", "cl", "-cl-std=CL3.0", "-Werror", "-fsyntax-only", str(generated)], check=True)
        print("PASS: subgroup synchronization retains memory fences and subgroup scope")

        source = work / "numeric-record.ll"
        output, generated = work / "numeric-record.mcl", work / "numeric-record.cl"
        source.write_text(preamble.replace("e-p:64:64", "e-p:64:64-i64:64") + """
define internal {float, i64, float, i64} @record(float %v, i64 %idx) noinline {
 %a = insertvalue {float, i64, float, i64} undef, float %v, 0
 %b = insertvalue {float, i64, float, i64} %a, i64 %idx, 1
 %c = insertvalue {float, i64, float, i64} %b, float 2.0, 2
 %d = insertvalue {float, i64, float, i64} %c, i64 3, 3
 ret {float, i64, float, i64} %d
}
define void @numeric_record(ptr %out, float %v, i64 %idx, i1 %choose) {
 %a = call {float, i64, float, i64} @record(float %v, i64 %idx)
 %b = select i1 %choose, {float, i64, float, i64} %a, {float, i64, float, i64} zeroinitializer
 store {float, i64, float, i64} %b, ptr %out, align 8
 %field = extractvalue {float, i64, float, i64} %b, 2
 %tail = getelementptr i8, ptr %out, i64 32
 store float %field, ptr %tail, align 4
 ret void
}
define internal <{i8, i64}> @packed_record(i8 %small, i64 %large) noinline {
 %a = insertvalue <{i8, i64}> undef, i8 %small, 0
 %b = insertvalue <{i8, i64}> %a, i64 %large, 1
 ret <{i8, i64}> %b
}
define void @packed_numeric_record(ptr %out, i8 %small, i64 %large) {
 %a = call <{i8, i64}> @packed_record(i8 %small, i64 %large)
 store <{i8, i64}> %a, ptr %out, align 1
 ret void
}
""")
        subprocess.run([emitter, str(source), str(output), str(generated), "--mark-exported"], check=True)
        text = generated.read_text()
        assert "packed,aligned(8)" in text and "packed,aligned(1)" in text
        assert "_size_check[sizeof(" in text and "field_3" in text
        if len(sys.argv) > 2:
            subprocess.run([sys.argv[2], "--target=spir64-unknown-unknown", "-x", "cl", "-cl-std=CL1.2", "-Werror", "-fsyntax-only", str(generated)], check=True)
        print("PASS: numeric reduction records preserve natural and packed layouts")

        # Failure transport is an implicit backend argument, propagated through
        # helper calls. A failed helper must return before its caller's store.
        source = work / "assertions.ll"
        output, generated = work / "assertions.mcl", work / "assertions.cl"
        source.write_text(preamble + """
declare void @llvm.trap()
define internal i32 @checked_value(i32 %value) noinline {
 %ok = icmp sgt i32 %value, 0
 br i1 %ok, label %good, label %bad
good:
 ret i32 %value
bad:
 call void @llvm.trap()
 unreachable
}
define void @checked(ptr %out, i32 %value) {
 %result = call i32 @checked_value(i32 %value)
 store i32 %result, ptr %out
 ret void
}
""")
        subprocess.run([emitter, str(source), str(output), str(generated), "--mark-exported"], check=True)
        text = generated.read_text()
        assert "__global uint *mojo_error" in text and "atomic_or(" in text
        assert "__builtin_trap" not in text
        data = output.read_bytes()
        assert data[:8] == b"MOJOCL03"
        cursor = 8
        def assertion_u32() -> int:
            nonlocal cursor
            value = int.from_bytes(data[cursor:cursor + 4], "little")
            cursor += 4
            return value
        assert assertion_u32() == 120 and assertion_u32() == 0 and assertion_u32() == 1
        for _ in range(2):
            length = assertion_u32()
            cursor += length
        assert assertion_u32() == 2  # implicit device-error buffer, no shared memory
        if len(sys.argv) > 2:
            subprocess.run([sys.argv[2], "--target=spir64-unknown-unknown", "-x", "cl", "-cl-std=CL1.2", "-Werror", "-fsyntax-only", str(generated)], check=True)
        print("PASS: device errors propagate through helpers and manifest flags")

        # A multi-kernel artifact must preserve source symbol lookup while
        # independently describing each native signature and local allocation.
        source = work / "shared.ll"
        output, generated = work / "shared.mcl", work / "shared.cl"
        source.write_text(
            preamble
            + """
@local = internal addrspace(3) global [8 x float] undef, align 4
@dynamic = external addrspace(3) global [0 x i8], align 4
define void @with_local(ptr %out) {
 %p = getelementptr [8 x float], ptr addrspace(3) @local, i64 0, i64 0
 store float 1.0, ptr addrspace(3) %p, align 4
 %v = load float, ptr addrspace(3) %p, align 4
 store float %v, ptr %out, align 4
 ret void
}
define void @with_dynamic(ptr %out) {
 %p = getelementptr [0 x i8], ptr addrspace(3) @dynamic, i64 0, i64 0
 store float 2.0, ptr addrspace(3) %p, align 4
 %v = load float, ptr addrspace(3) %p, align 4
 store float %v, ptr %out, align 4
 ret void
}
define void @"ordinary.mangled<kernel>"(ptr %out) {
 store float 3.0, ptr %out, align 4
 ret void
}
"""
        )
        subprocess.run(
            [
                emitter,
                str(source),
                str(output),
                str(generated),
                "--mark-exported",
            ],
            check=True,
        )
        data, cursor = output.read_bytes(), 8
        assert data[:8] == b"MOJOCL03"

        def u32() -> int:
            nonlocal cursor
            result = int.from_bytes(data[cursor : cursor + 4], "little")
            cursor += 4
            return result

        def string() -> str:
            nonlocal cursor
            length = u32()
            value = data[cursor : cursor + length].decode()
            cursor += length
            return value

        assert u32() == 120  # OpenCL C version
        assert u32() == 0  # no optional extensions
        assert u32() == 3
        signatures = {}
        for _ in range(3):
            name, native = string(), string()
            dynamic, subgroup_size = u32(), u32()
            arguments = []
            for _ in range(u32()):
                kind, size, fields = u32(), u32(), u32()
                arguments.append(
                    (kind, size, [(u32(), u32(), u32()) for _ in range(fields)])
                )
            signatures[name] = (native, dynamic)
            assert subgroup_size == 0 and arguments == [(4, 32, [(0, 3, 24)])]
        assert signatures["with_local"][1] == 0
        assert signatures["with_dynamic"][1] == 1
        assert signatures["ordinary.mangled<kernel>"][1] == 0
        assert len({v[0] for v in signatures.values()}) == 3
        print("PASS: versioned mangled symbols and per-kernel local-memory ABI")

        # Typed aggregate and borrowed aggregate pointers share one logical
        # manifest boundary. Native parameters retain pointer provenance.
        source.write_text(
            preamble
            + """
define void @aggregate({ptr,{i32,ptr},i1} %x, float %bias) {
 %p = extractvalue {ptr,{i32,ptr},i1} %x, 0
 %out = extractvalue {ptr,{i32,ptr},i1} %x, 1, 1
 %bits = ptrtoint ptr %p to i64
 %qualified = inttoptr i64 %bits to ptr addrspace(1)
 %v = load float, ptr addrspace(1) %qualified
 %r = fadd float %v, %bias
 %flag = extractvalue {ptr,{i32,ptr},i1} %x, 2
 %chosen = select i1 %flag, float %r, float %bias
 store float %chosen, ptr %out
 ret void
}
define void @borrowed(ptr byval({ptr,{i32,ptr}}) %x) {
 %field = getelementptr {ptr,{i32,ptr}}, ptr %x, i32 0, i32 1, i32 1
 %out = load ptr, ptr %field
 store float 4.0, ptr %out
 ret void
}
"""
        )
        subprocess.run(
            [
                emitter,
                str(source),
                str(output),
                str(generated),
                "--mark-exported",
            ],
            check=True,
        )
        data, cursor = output.read_bytes(), 8
        assert data[:8] == b"MOJOCL03"
        assert u32() == 120 and u32() == 0 and u32() == 2
        aggregate_signatures = {}
        for _ in range(2):
            name, native = string(), string()
            assert u32() == 0 and u32() == 0
            aggregate_arguments = []
            for _ in range(u32()):
                kind, size = u32(), u32()
                assert kind == 4
                aggregate_fields = [(u32(), u32(), u32()) for _ in range(u32())]
                aggregate_arguments.append((size, aggregate_fields))
            aggregate_signatures[name] = aggregate_arguments
        view_fields = [(0, 3, 24), (8, 2, 4), (16, 3, 24)]
        aggregate_fields_with_bool = [*view_fields, (24, 2, 1)]
        assert aggregate_signatures["aggregate"] == [
            (128, aggregate_fields_with_bool),
            (32, [(0, 2, 4)]),
        ]
        assert aggregate_signatures["borrowed"] == [(96, view_fields)]
        assert "unsupported" not in generated.read_text()
        print(
            "PASS: typed nested/byval aggregate manifests preserve pointer leaves"
        )

        source.write_text(
            preamble
            + """
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr>, i32, <4 x i1>, <4 x float>)
declare void @llvm.masked.scatter.v4f32.v4p0(<4 x float>, <4 x ptr>, i32, <4 x i1>)
declare <4 x i64> @llvm.stepvector.v4i64()
declare <2 x half> @llvm.fma.v2f16(<2 x half>, <2 x half>, <2 x half>)
define internal void @read_pointer(ptr addrspace(5) %slot, ptr %out) noinline {
 %p = load ptr addrspace(1), ptr addrspace(5) %slot, align 8
 %v = load float, ptr addrspace(1) %p, align 4
 store float %v, ptr %out, align 4
 ret void
}
define void @coverage(ptr %out, ptr %counter, ptr %wide, ptr %total, i64 %idx, i64 %scratch_index, <2 x half> %halfA, <2 x half> %halfB) #0 {
 %pointer_slot = alloca ptr addrspace(1), align 8, addrspace(5)
 %global_pointer = addrspacecast ptr %out to ptr addrspace(1)
 store ptr addrspace(1) %global_pointer, ptr addrspace(5) %pointer_slot, align 8
 call void @read_pointer(ptr addrspace(5) %pointer_slot, ptr %counter)
 %wrapped = insertvalue {i64} poison, i64 %idx, 0
 %stable = freeze {i64} %wrapped
 %stable_index = extractvalue {i64} %stable, 0
 %flag = trunc i64 %stable_index to i1
 %mask0 = insertelement <4 x i1> poison, i1 %flag, i32 0
 %mask = shufflevector <4 x i1> %mask0, <4 x i1> poison, <4 x i32> zeroinitializer
 %indices = call <4 x i64> @llvm.stepvector.v4i64()
 %srcs = getelementptr float, ptr %out, <4 x i64> %indices
 %address = ptrtoint ptr %total to i64
 %address0 = insertelement <4 x i64> poison, i64 %address, i32 0
 %addresses = shufflevector <4 x i64> %address0, <4 x i64> poison, <4 x i32> zeroinitializer
 %byte_offsets = mul <4 x i64> %indices, <i64 4, i64 4, i64 4, i64 4>
 %address_values = add <4 x i64> %addresses, %byte_offsets
 %dsts = inttoptr <4 x i64> %address_values to <4 x ptr>
 %gathered = call <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr> %srcs, i32 4, <4 x i1> %mask, <4 x float> zeroinitializer)
 call void @llvm.masked.scatter.v4f32.v4p0(<4 x float> %gathered, <4 x ptr> %dsts, i32 4, <4 x i1> %mask)
 %unaligned = load float, ptr %out, align 1
 store float %unaligned, ptr %counter, align 1
 %unaligned_vector = load <3 x i64>, ptr %wide, align 1
 store <3 x i64> %unaligned_vector, ptr %out, align 1
 %h = call <2 x half> @llvm.fma.v2f16(<2 x half> %halfA, <2 x half> %halfB, <2 x half> %halfB)
 %fast = call afn <2 x half> @llvm.fma.v2f16(<2 x half> %halfA, <2 x half> %halfB, <2 x half> %halfB)
 store <2 x half> %h, ptr %out
 store <2 x half> %fast, ptr %counter
 %scratch = alloca [8 x i32], align 16
 call void @llvm.memset.p0.i64(ptr %scratch, i8 0, i64 32, i1 false)
 %slot = getelementptr [8 x i32], ptr %scratch, i64 0, i64 %idx
 store i32 7, ptr %slot
 %read_slot = getelementptr [8 x i32], ptr %scratch, i64 0, i64 %scratch_index
 %n = load i32, ptr %read_slot
 store i32 %n, ptr %out
 %cas = cmpxchg ptr %counter, i32 0, i32 1 acq_rel acquire
 %old = extractvalue {i32,i1} %cas, 0
 %success = extractvalue {i32,i1} %cas, 1
 %old64 = atomicrmw add ptr %wide, i64 1 monotonic, align 8
 store atomic float 1.0, ptr %total release, align 4
 %loaded = load atomic float, ptr %total acquire, align 4
 %sum = atomicrmw fadd ptr %total, float 0.5 monotonic, align 4
 switch i64 %idx, label %other [i64 0, label %first i64 1, label %second]
first:
 br label %join
second:
 br label %join
other:
 br label %join
join:
 %chosen = phi i32 [7, %first], [11, %second], [13, %other]
 %joined = phi {i64} [%wrapped, %first], [zeroinitializer, %second], [zeroinitializer, %other]
 %joined_stable = freeze {i64} %joined
 %joined_value = extractvalue {i64} %joined_stable, 0
 store i64 %joined_value, ptr %wide
 store i32 %chosen, ptr %counter
 ret void
}
attributes #0 = { "mojo.opencl.work_group_size_hint"="128" }
"""
        )
        subprocess.run(
            [
                emitter,
                str(source),
                str(output),
                str(generated),
                "--mark-exported",
            ],
            check=True,
        )
        text = generated.read_text()
        assert "__global uchar * __private *" in text
        assert "atomic_compare_exchange_strong_explicit" in text
        assert "atomic_ulong" in text and "cl_khr_int64_base_atomics" in text
        assert "__private uchar" in text and "_storage[32]" in text
        assert "work_group_size_hint(128,1,1)" in text
        assert "switch (" in text and "default: {" in text
        assert "as_float(vload4(" in text
        assert "as_ulong(vload8(" in text
        assert "vstore8(as_uchar8(" in text
        assert "as_float(" in text
        assert " = fma(" in text and " = mad(" in text
        print(
            "PASS: private stack provenance, memory intrinsic, compare-exchange, float CAS and i64 atomics"
        )
        if len(sys.argv) > 2:
            subprocess.run(
                [
                    sys.argv[2],
                    "--target=spir64-unknown-unknown",
                    "-x",
                    "cl",
                    "-cl-std=CL3.0",
                    "-Werror",
                    "-fsyntax-only",
                    str(generated),
                ],
                check=True,
            )
            print("PASS: generated coverage source compiles as OpenCL C 3.0")


if __name__ == "__main__":
    main()
