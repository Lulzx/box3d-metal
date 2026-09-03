#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Box3D Metal contributors
# SPDX-License-Identifier: MIT
"""Phase 0 roofline evaluator for Metal solver throughput.

Reads one row of metal_scene_benchmark / metal_contact_benchmark style CSV
(contact count, analytic solver bytes, step milliseconds, achievable GB/s from
metal_bandwidth) and reports where the step sits relative to the bandwidth
roofline.

Usage:
    python3 tools/metal_roofline.py --contacts 65536 --bytes 23592960 \\
        --step-ms 10.5 --bandwidth-gbs 200
    python3 tools/metal_roofline.py --csv row.csv  # contacts,bytes,step_ms,bandwidth_gbs header
"""

import argparse
import csv
import sys


def evaluate(contacts: int, byte_count: int, step_ms: float, bandwidth_gbs: float) -> dict:
    if step_ms <= 0 or bandwidth_gbs <= 0:
        raise ValueError("step_ms and bandwidth_gbs must be positive")
    seconds = step_ms / 1000.0
    achieved_gbs = byte_count / seconds / 1e9
    floor_ms = byte_count / (bandwidth_gbs * 1e9) * 1000.0
    return {
        "contacts": contacts,
        "bytes_per_contact_step": byte_count / contacts if contacts > 0 else 0.0,
        "achieved_gbs": achieved_gbs,
        "bandwidth_gbs": bandwidth_gbs,
        "floor_ms": floor_ms,
        "step_ms": step_ms,
        "fraction_of_roofline": achieved_gbs / bandwidth_gbs,
        "headroom_vs_floor": step_ms / floor_ms if floor_ms > 0 else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=int, default=0)
    parser.add_argument("--bytes", type=int, default=0)
    parser.add_argument("--step-ms", type=float, default=0.0)
    parser.add_argument("--bandwidth-gbs", type=float, default=0.0)
    parser.add_argument("--csv", default=None,
                        help="CSV file with header contacts,bytes,step_ms,bandwidth_gbs")
    args = parser.parse_args()

    rows = []
    if args.csv is not None:
        with open(args.csv, newline="") as f:
            for row in csv.DictReader(f):
                rows.append((int(row["contacts"]), int(row["bytes"]),
                             float(row["step_ms"]), float(row["bandwidth_gbs"])))
    else:
        rows.append((args.contacts, args.bytes, args.step_ms, args.bandwidth_gbs))

    print("contacts,bytes,bytes_per_contact,step_ms,floor_ms,achieved_gbs,roofline_gbs,"
          "fraction_of_roofline,headroom_vs_floor")
    for contacts, byte_count, step_ms, bandwidth_gbs in rows:
        try:
            r = evaluate(contacts, byte_count, step_ms, bandwidth_gbs)
        except ValueError as e:
            print(f"skip row contacts={contacts}: {e}", file=sys.stderr)
            continue
        print(f"{r['contacts']},{byte_count},{r['bytes_per_contact_step']:.1f},"
              f"{r['step_ms']:.3f},{r['floor_ms']:.3f},{r['achieved_gbs']:.2f},"
              f"{r['bandwidth_gbs']:.2f},{r['fraction_of_roofline']:.3f},"
              f"{r['headroom_vs_floor']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
