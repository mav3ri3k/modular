"""Wrapper for mojo_shared_library to add internal logic."""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")
load("@rules_mojo//mojo:mojo_shared_library.bzl", _upstream_mojo_shared_library = "mojo_shared_library")

# A variant of the upstream mojo_shared_library that, under `--config=asan`,
# runs the production (non-instrumented) compiler -- fast -- while the target
# itself and its dependencies stay in the incoming asan configuration. The
# transition sits on the compiler attribute rather than on the rule; see
# mojo_library.bzl for why, and cc_transition.bzl for the asan-only gating
# (`new_config`).
_transitioned_mojo_shared_library = rule(
    implementation = lambda ctx: ctx.super(),
    parent = _upstream_mojo_shared_library,
    attrs = {
        # Must stay in sync with the toolchain's `mojo`; this is the same
        # compiler, first selected for the build machine, then reconfigured.
        "_mojo_override": attr.label(
            default = Label("//bazel/internal:production_mojo_compiler"),
            executable = True,
            cfg = CFG_WORKAROUND,
        ),
    },
)

def mojo_shared_library(
        name,
        use_production_compiler_for_asan = False,
        tags = [],
        **kwargs):
    library_rule = _transitioned_mojo_shared_library if use_production_compiler_for_asan else _upstream_mojo_shared_library
    library_rule(
        name = name,
        tags = ["mojo-fixits"] + tags,
        **kwargs
    )
