#!/usr/bin/env python3
"""
Score a retrieval traces JSONL (from script/run.py) and write an
e2e_phase5_metrics.json compatible with create_fair_trackb2_growth_figures.py.

Used by Phase 6 T1.4 to measure retrieval impact of OpenCV-injected
position_context on the G8 precomputed trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rank_from_trace(row: dict) -> tuple[int | None, int | None]:
    """Return (rank_1_based, pool_size). rank=None means GT not in returned candidates."""
    io = row.get("interpreter_output") or {}
    cands = io.get("candidates") or []
    gt_guid = (row.get("scenario") or {}).get("ground_truth", {}).get("target_guid")
    pool = row.get("final_pool_size")
    if not gt_guid or not cands:
        return None, pool
    for i, c in enumerate(cands, start=1):
        if c.get("guid") == gt_guid:
            return i, pool
    return None, pool


def score(traces_path: Path) -> dict[str, Any]:
    rows: list[dict] = []
    with traces_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    n = len(rows)
    gt_in_pool = 0
    top1 = top5 = top10 = 0
    mrr_sum = 0.0
    pool_sizes: list[int] = []

    for row in rows:
        rank, pool = _rank_from_trace(row)
        # Rank is 1-based position of GT within the returned candidates list
        # (which is typically truncated to top-10). rank=None means GT is not in
        # the candidates — for the Track B-2 comparison we treat this as
        # "not in pool" for Top-K purposes. NOTE: the pipeline's `guid_match`
        # field means "GT in candidates" (≈ gt-in-pool), NOT rank=1.
        if rank is not None:
            gt_in_pool += 1
            if rank == 1:
                top1 += 1
            if rank <= 5:
                top5 += 1
            if rank <= 10:
                top10 += 1
            mrr_sum += 1.0 / rank
        if pool:
            pool_sizes.append(int(pool))

    avg_pool = sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0.0
    metrics = {
        "traces_path": str(traces_path),
        "overall": {
            "n": n,
            "gt_in_pool": gt_in_pool,
            "gt_in_pct": 100.0 * gt_in_pool / n if n else 0.0,
            "top1": top1,
            "top1_pct": 100.0 * top1 / n if n else 0.0,
            "top5": top5,
            "top5_pct": 100.0 * top5 / n if n else 0.0,
            "top10": top10,
            "top10_pct": 100.0 * top10 / n if n else 0.0,
            "mrr": round(mrr_sum / n, 4) if n else 0.0,
            "avg_pool": round(avg_pool, 1),
        },
    }
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("traces_path", type=Path, help="traces_*.jsonl from script/run.py")
    p.add_argument("--out", type=Path, required=True, help="Output metrics JSON path")
    args = p.parse_args()

    metrics = score(args.traces_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["overall"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
