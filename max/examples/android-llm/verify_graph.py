#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
"""Compare every MAX graph logit and cache value with graph_reference.py."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--max-absolute-error", type=float, default=0.05)
    parser.add_argument("--max-relative-l2-error", type=float, default=0.005)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {}
    passed = True
    for i, name in enumerate(("logits", "cache")):
        actual_path = args.actual / f"output{i}"
        expected_path = args.reference / f"expected{i}"
        if actual_path.stat().st_size != expected_path.stat().st_size:
            raise ValueError(f"{name}: actual and reference sizes differ")
        actual = np.fromfile(actual_path, dtype="<f4").astype(np.float64)
        expected = np.fromfile(expected_path, dtype="<f4").astype(np.float64)
        if not actual.size or not np.isfinite(expected).all():
            raise ValueError(f"{name}: empty or nonfinite reference")
        finite = bool(np.isfinite(actual).all())
        difference = actual - expected
        absolute = float(np.max(np.abs(difference))) if finite else None
        relative = float(
            np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)
        ) if finite else None
        valid = (
            finite
            and absolute <= args.max_absolute_error
            and relative <= args.max_relative_l2_error
        )
        result = dict(
            elements=actual.size,
            finite=finite,
            max_absolute_error=absolute,
            relative_l2_error=relative,
        )
        if i == 0:
            result["greedy_id"] = int(actual.argmax()) if finite else None
            result["reference_greedy_id"] = int(expected.argmax())
            valid = (
                valid and result["greedy_id"] == result["reference_greedy_id"]
            )
        result["passed"] = bool(valid)
        passed = passed and valid
        report[name] = result
    report["passed"] = bool(passed)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
