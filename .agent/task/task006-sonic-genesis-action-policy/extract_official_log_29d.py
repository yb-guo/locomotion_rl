"""Extract 29D motor columns from official SONIC deploy CSV logs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--cols", type=int, default=29)
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()

    rows = extract_tail_columns(Path(args.src), cols=args.cols, rows=args.rows)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)

    flat = [value for row in rows for value in row]
    print("SRC", args.src)
    print("DST", args.dst)
    print("ROWS", len(rows))
    print("COLS", args.cols)
    print("FINITE", all(math.isfinite(value) for value in flat))
    print("MIN_MAX", min(flat), max(flat))
    print("ROW0_ABS_MAX", max(abs(value) for value in rows[0]))


def extract_tail_columns(path: Path, *, cols: int, rows: int) -> list[list[float]]:
    if cols <= 0:
        raise ValueError("cols must be positive")
    if rows <= 0:
        raise ValueError("rows must be positive")
    extracted: list[list[float]] = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) < cols:
            raise ValueError(f"{path} has {len(header)} columns, cannot extract {cols}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) < cols:
                raise ValueError(f"{path}:{row_number} has {len(row)} columns")
            extracted.append([float(value) for value in row[-cols:]])
            if len(extracted) >= rows:
                break
    if not extracted:
        raise ValueError(f"{path} has no data rows")
    return extracted


if __name__ == "__main__":
    main()
