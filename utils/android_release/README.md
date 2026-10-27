# Publishing the Android fork

The public branch should contain one sanitized Android snapshot on top of
upstream commit `f85b9f5e`, rather than the original five development commits.
This keeps the upstream ancestry and GitHub fork relationship while excluding
the development commits that contain device identifiers. Create the repository
using GitHub's **Fork** button, then push only the publication branch. Do not
push `--all`, `--mirror`, tags pointing into the development history, or the
original Android branch.

## Privacy filtering

From the repository root:

```sh
python3 utils/android_release/sanitize_publication.py --write
python3 utils/android_release/sanitize_publication.py
```

The tool examines tracked changes relative to the upstream base and untracked,
nonignored files. It replaces literal device serials found in metadata and adb
commands, personal home-directory names, MAC addresses, and IPv4 addresses.
Battery/thermal dumps are reduced to an allowlist of battery level, charging
state, temperature, voltage, health, and thermal status. Battery identifiers,
event history, usage history, manufacturing/first-use dates, service listeners,
and unrecognized fields are discarded. JSON is parsed so nested snapshot
strings can be filtered without corrupting the file structure.

Benchmark rates, durations, model revisions, artifact hashes, GPU names, and
firmware versions remain available. Published logs are privacy-filtered copies;
hashes recorded during measurement describe the original measured artifacts,
not newly sanitized log or source-snapshot bytes. The archived CMake cache has
redacted paths and is evidence of settings, not an executable build recipe.

This is conservative filtering, not a proof that arbitrary future files contain
no private information. Review new files before each release. The tool does not
modify `.git`, ignored local files, or upstream history, and it is not a general
credential scanner. Git author names/emails are ordinary public attribution and
are not removed.

## Source scope

The Android work includes compiler changes as well as runtime changes:

- New OpenCL target traits, lowering, and a roughly 2,000-line source emitter.
- Shared compiler hooks for target emission, C ABI selection, and lifetime
  checking of associated storage fields. The ABI/host-target exceptions are
  largely conditional on OpenCL; the lifetime-checking changes are shared.
- A process-wide CPU worker-pool reuse change in the Mojo runtime.
- HAL device/runtime integration and opaque tensor storage support. Changes to
  `TileTensor`, `TensorEngine`, and their callers also touch existing GPU code.

Keep the experimental scope explicit. Targeted regression coverage exists in
the OpenCL compiler tests, associated-storage parser test, HAL device tests,
opaque tensor/buffer tests, and graph-storage tests. Android benchmark passes
alone do not establish regression coverage for every existing backend.

## Release checks

1. Run the privacy filter and inspect the complete proposed snapshot.
2. Confirm pinned host dependencies and the reference build instructions.
3. Build and validate the chosen models from the intended publication tree.
4. Create a signed-off snapshot commit whose parent is the upstream base,
   including an `Assisted-by: AI` trailer. Preserve the original local branch.
5. Check the new branch's history before pushing only that branch to the fork.

The snapshot commit intentionally does not preserve the Android development
commit sequence. Keep that sequence locally if useful; publishing it later
would require a separate history-wide sanitization audit.
