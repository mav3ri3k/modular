# Galaxy S26 Bonsai 2 fresh rebuild

This directory records a fresh build and rerun on the Galaxy S26 SM-S942U1
(SM8850 / Adreno 840, Android API 37) on October 26, 2026. It used source
revision `378087fb5fec89841ea7d5e3ac3cc580964ee884` with the pre-existing
worktree edits recorded in the session, Python 3.13.11, NDK 29.0.14206865 and
the macOS 26.5 SDK override.

`benchmark-summary.json` contains the three separate-process `tg128` and
`pp512` rates. Matching `*-*.stdout` and `*-*.stderr` files preserve each
result, and `*-thermal-before/after.txt` files preserve the device state around
each run. `validate-{2,4,9}.log` records the on-device correctness checks;
`parity{2,4,9}.json` records the exporter checks against the pinned GGUFs.
`provenance.json` lists input and binary hashes. The source worktree had
pre-existing local edits; these were left intact.

These were grouped by model with the README's ten-second pauses. Thermal status
rose during the larger-model runs. The fresh `pp512` Mojo results are not paired
with the Prism rates in the earlier reports. Archived binaries also have
several different hashes, so this rerun documents the fresh repository build,
not a byte-identical rebuild of the archived benchmark binaries.
