"""Expose the selected Android NDK sysroot and compiler runtime to C++ builds."""

def _android_ndk_repository_impl(ctx):
    ndk_home = ctx.getenv("ANDROID_NDK_HOME")
    if not ndk_home:
        # Non-Android builds must not require an NDK installation.
        ctx.file("BUILD.bazel", "")
        return
    ndk = ctx.path(ndk_home)
    if not ndk.get_child("source.properties").exists:
        fail("ANDROID_NDK_HOME must name an Android NDK installation")
    host = "darwin-x86_64" if ctx.os.name == "mac os x" else "linux-x86_64"
    prebuilt = ndk.get_child("toolchains", "llvm", "prebuilt", host)
    if not prebuilt.exists:
        fail("The NDK does not contain tools for " + host)
    versions = prebuilt.get_child("lib", "clang").readdir()
    if len(versions) != 1:
        fail("Expected one Clang resource directory in the NDK")

    # The directory provider carries a root as well as its files, matching the
    # other sysroots consumed by the shared C++ toolchain.
    for part, source in [("sysroot", prebuilt.get_child("sysroot")), ("resource", versions[0])]:
        for child in source.readdir():
            if child.basename != "BUILD.bazel":
                ctx.symlink(child, part + "/" + child.basename)
        ctx.file(part + "/BUILD.bazel", """
load("@bazel_skylib//rules/directory:directory.bzl", "directory")
package(default_visibility = ["//visibility:public"])
directory(name = "root", srcs = glob(["**"]))
filegroup(name = "files", srcs = glob(["**"]))
""")
    for child in prebuilt.readdir():
        if child.basename != "BUILD.bazel":
            ctx.symlink(child, "toolchain/" + child.basename)
    ctx.file("toolchain/BUILD.bazel", """
package(default_visibility = ["//visibility:public"])
exports_files(["bin/clang", "bin/clang++", "bin/llvm-ar", "bin/llvm-strip", "bin/llvm-objcopy"])
filegroup(name = "runtime", srcs = glob(["bin/clang*", "bin/ld*", "bin/lld*", "lib/**", "lib64/**"], allow_empty = True))
""")
    ctx.file("BUILD.bazel", "")

android_ndk_repository = repository_rule(
    implementation = _android_ndk_repository_impl,
    environ = ["ANDROID_NDK_HOME"],
    local = True,
)
