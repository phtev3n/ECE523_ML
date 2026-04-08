"""Parse Trackman CSV export into per-shot launch condition dicts.

Usage
-----
python scripts/parse_trackman.py \
    --csv trackman_session.csv \
    --out trackman_shots.json

The script handles common Trackman CSV column names and converts
imperial units to metric when detected.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Mapping of common Trackman column names → normalised keys.
# Trackman exports vary by software version; these cover the most common ones.
COLUMN_MAP = {
    # Ball speed
    "ball speed": "ball_speed",
    "ball speed (m/s)": "ball_speed",
    "ball speed (mph)": "ball_speed_mph",
    "ballspeed": "ball_speed",
    # Launch angle
    "launch angle": "launch_angle_deg",
    "launch angle (deg)": "launch_angle_deg",
    "launchangle": "launch_angle_deg",
    "v launch angle": "launch_angle_deg",
    # Launch direction (azimuth)
    "launch direction": "launch_direction_deg",
    "launch direction (deg)": "launch_direction_deg",
    "h launch angle": "launch_direction_deg",
    "launchdirection": "launch_direction_deg",
    # Spin
    "spin rate": "spin_rate_rpm",
    "spin rate (rpm)": "spin_rate_rpm",
    "spinrate": "spin_rate_rpm",
    "total spin": "spin_rate_rpm",
    "spin axis": "spin_axis_deg",
    "spin axis (deg)": "spin_axis_deg",
    "spinaxis": "spin_axis_deg",
    # Carry
    "carry": "carry_m",
    "carry (m)": "carry_m",
    "carry (yds)": "carry_yds",
    "carry distance": "carry_m",
    "carry dist": "carry_m",
    "carry dist.": "carry_m",
    # Apex
    "apex height": "apex_m",
    "apex height (m)": "apex_m",
    "apex height (ft)": "apex_ft",
    "max height": "apex_m",
    "max height (m)": "apex_m",
    "max height (ft)": "apex_ft",
    # Landing angle
    "landing angle": "landing_angle_deg",
    "landing angle (deg)": "landing_angle_deg",
    "land angle": "landing_angle_deg",
    # Club info (optional but useful)
    "club speed": "club_speed",
    "club speed (m/s)": "club_speed",
    "club speed (mph)": "club_speed_mph",
    "club": "club_name",
    "club type": "club_name",
    # Shot ID
    "shot": "shot_number",
    "shot #": "shot_number",
    "shot number": "shot_number",
    "no.": "shot_number",
    "#": "shot_number",
}

MPH_TO_MS = 0.44704
YDS_TO_M = 0.9144
FT_TO_M = 0.3048


def normalise_header(header: str) -> str:
    """Lowercase and strip whitespace/BOM from a CSV header."""
    return re.sub(r"[^\w\s/().#]", "", header.strip().lower().strip("\ufeff"))


def parse_float(val: str) -> float | None:
    """Try to parse a numeric value, returning None on failure."""
    val = val.strip().replace(",", "")
    if val in ("", "-", "--", "N/A", "n/a"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def convert_units(shot: dict) -> dict:
    """Convert any imperial-unit fields to metric."""
    if "ball_speed_mph" in shot and "ball_speed" not in shot:
        v = shot.pop("ball_speed_mph")
        if v is not None:
            shot["ball_speed"] = v * MPH_TO_MS
    if "carry_yds" in shot and "carry_m" not in shot:
        v = shot.pop("carry_yds")
        if v is not None:
            shot["carry_m"] = v * YDS_TO_M
    if "apex_ft" in shot and "apex_m" not in shot:
        v = shot.pop("apex_ft")
        if v is not None:
            shot["apex_m"] = v * FT_TO_M
    if "club_speed_mph" in shot and "club_speed" not in shot:
        v = shot.pop("club_speed_mph")
        if v is not None:
            shot["club_speed"] = v * MPH_TO_MS
    return shot


def parse_trackman_csv(csv_path: Path) -> list[dict]:
    """Read a Trackman CSV and return a list of per-shot dicts with metric units."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        norm_headers = [normalise_header(h) for h in raw_headers]

        # Map each column index to our normalised key
        col_map: dict[int, str] = {}
        for i, nh in enumerate(norm_headers):
            if nh in COLUMN_MAP:
                col_map[i] = COLUMN_MAP[nh]

        if not col_map:
            sys.exit(
                f"Could not match any known Trackman columns.\n"
                f"Headers found: {raw_headers}\n"
                f"Normalised:    {norm_headers}"
            )

        shots = []
        for row_num, row in enumerate(reader, start=2):
            shot: dict = {}
            for col_idx, key in col_map.items():
                if col_idx < len(row):
                    val = parse_float(row[col_idx])
                    if val is not None:
                        shot[key] = val
                    elif key in ("club_name", "shot_number"):
                        shot[key] = row[col_idx].strip()
            if not shot:
                continue
            shot = convert_units(shot)
            shot["_csv_row"] = row_num
            shots.append(shot)

    return shots


def validate_shots(shots: list[dict]) -> list[dict]:
    """Filter out obvious Trackman misreads."""
    valid = []
    for s in shots:
        bs = s.get("ball_speed")
        la = s.get("launch_angle_deg")
        carry = s.get("carry_m")
        if bs is not None and (bs < 5 or bs > 100):
            print(f"  WARNING: row {s.get('_csv_row')}: ball_speed={bs:.1f} m/s out of range, skipping")
            continue
        if la is not None and (la < 0 or la > 60):
            print(f"  WARNING: row {s.get('_csv_row')}: launch_angle={la:.1f}° out of range, skipping")
            continue
        if carry is not None and carry < 1:
            print(f"  WARNING: row {s.get('_csv_row')}: carry={carry:.1f}m too short, skipping")
            continue
        valid.append(s)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Trackman CSV export into per-shot JSON")
    parser.add_argument("--csv", type=str, required=True, help="Trackman CSV file path")
    parser.add_argument("--out", type=str, default="trackman_shots.json", help="Output JSON path")
    parser.add_argument("--no_validate", action="store_true", help="Skip validation filtering")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    shots = parse_trackman_csv(csv_path)
    print(f"Parsed {len(shots)} shots from {csv_path}")

    if not args.no_validate:
        shots = validate_shots(shots)
        print(f"After validation: {len(shots)} shots")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(shots, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
