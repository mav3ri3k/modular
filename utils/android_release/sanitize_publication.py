#!/usr/bin/env python3
"""Conservatively redact Android fork additions relative to an upstream commit.

This checks the working tree, not Git history. Publish a snapshot with the
upstream commit as parent; do not push the original development history.
"""

import argparse
import ipaddress
import json
import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args], text=True).splitlines()


def thermal_summary(text):
    # Deliberately discard event history, listener/binder IDs, battery QR codes,
    # manufacturing/first-use dates, usage history, and unrecognized fields.
    keep = []
    for line in text.splitlines():
        if re.fullmatch(
            r"\s*(?:(?:AC|USB|Wireless|Dock) powered: (?:true|false)"
            r"|(?:level|scale|temperature|voltage|status|health): -?\d+"
            r"|(?:Current )?Thermal Status: \d+)\s*", line
        ):
            keep.append(line.strip())
    return "[Privacy-filtered snapshot; only battery/thermal summary retained]\n" + "\n".join(keep) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="f85b9f5e")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    paths = sorted(set(git("diff", "--name-only", args.base) + git("ls-files", "--others", "--exclude-standard")))
    texts = {}
    for name in paths:
        p = root / name
        if p.is_file() and name != str(Path(__file__).resolve().relative_to(root)):
            try:
                texts[name] = p.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                raise SystemExit(f"Review non-text file manually: {name}")
    # Learn literal serials from metadata and executed adb commands, then remove
    # every occurrence, including copies embedded in prose and command journals.
    serials = set()
    for text in texts.values():
        serials.update(re.findall(r'"(?:adb_)?serial"\s*:\s*"([A-Za-z0-9._:-]+)"', text))
        serials.update(re.findall(r'\badb\s+-s\s+[\"\x27]?([A-Za-z0-9._:-]{6,})', text))
        serials.update(re.findall(r"\badb: device ['\"]([^'\"]+)['\"]", text))
    serials -= {"REDACTED", "DEVICE_SERIAL", "SERIAL", "redacted"}

    def redact(text):
        for value in sorted(serials, key=len, reverse=True):
            text = text.replace(value, "DEVICE_SERIAL")
        text = re.sub(r"/(Users|home)/[^/\s\"'<>]+", r"/\1/REDACTED", text)
        text = re.sub(r"(?i)\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b", "REDACTED_MAC", text)
        def address(match):
            try:
                ipaddress.ip_address(match[0])
            except ValueError:
                return match[0]
            return "REDACTED_IP"
        text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", address, text)
        if "Current Battery Service state:" in text or "Current Thermal Service state:" in text:
            return thermal_summary(text)
        # For unstructured logs, drop entire sensitive lines, not just values.
        text = "\n".join(
            line for line in text.split("\n")
            if not re.search(r"(?i)(QrData|efsValue|FirstUseDate|IcAuthentication|android_id|imei|imsi|iccid|ro\.(?:boot\.)?serialno)\s*[:=]", line)
        )
        return text

    changed = []
    for name, original in texts.items():
        # Preserve JSON structure while filtering embedded device-state strings.
        if name.endswith(".json"):
            try:
                value = json.loads(original)
            except ValueError:
                updated = redact(original)
            else:
                def walk(value):
                    if isinstance(value, str):
                        return redact(value)
                    if isinstance(value, list):
                        return [walk(v) for v in value]
                    if isinstance(value, dict):
                        return {redact(k): walk(v) for k, v in value.items()}
                    return value
                clean = walk(value)
                updated = json.dumps(clean, indent=2) + "\n" if clean != value else original
        else:
            updated = redact(original)
        if updated != original:
            changed.append(name)
            if args.write:
                (root / name).write_text(updated)
    print(f"Scanned {len(texts)} files; {'sanitized' if args.write else 'would sanitize'} {len(changed)} files.")
    if changed and not args.write:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
