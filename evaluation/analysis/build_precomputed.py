#!/usr/bin/env python3
"""
Build precomputed constraints JSONL from cases_v3_test.jsonl ground truth labels.

This enables running the retrieval pipeline with oracle constraints (bypassing VLM),
which measures the upper bound of the retrieval pipeline's performance.

Usage:
  python eval/build_precomputed.py                    # All cases, output to eval/
  python eval/build_precomputed.py --spatial-only      # Only cases with spatial_relations
  python eval/build_precomputed.py --no-spatial         # Only cases WITHOUT spatial_relations

Output format (JSONL, one per line):
  {"case_id": "...", "status": "OK", "constraints": {...}}
"""

import json
import re
import sys
from pathlib import Path

CASES_FILE = Path(__file__).resolve().parent / "cases_v3_test.jsonl"
SKELETONS  = Path(__file__).resolve().parent.parent.parent / "data_curation" / "datasets" / "synth_v0.5" / "skeletons" / "skeletons_v2_5.jsonl"
OUTPUT     = Path(__file__).resolve().parent / "precomputed_oracle.jsonl"


def main():
    spatial_only = "--spatial-only" in sys.argv
    no_spatial = "--no-spatial" in sys.argv

    # Load skeletons for subject_type enrichment
    skeletons = {}
    with open(SKELETONS) as f:
        for line in f:
            sk = json.loads(line)
            skeletons[sk["id"]] = sk

    # Load cases
    with open(CASES_FILE) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    results = []
    fixes = 0
    for case in cases:
        constraints = dict(case["labels"]["constraints"])  # shallow copy
        spatial_rels = list(constraints.get("spatial_relations", []))

        if spatial_only and not spatial_rels:
            continue
        if no_spatial and spatial_rels:
            continue

        # Oracle enrichment from skeleton ground truth
        sk_match = re.search(r"SK_(\d+)", case["case_id"])
        sk_id = f"SK_{sk_match.group(1)}" if sk_match else None
        skel = skeletons.get(sk_id, {})
        target_props = skel.get("target_props", {})
        gt_type = target_props.get("Type", "")
        gt_storey = target_props.get("Storey", "")

        # Fix ifc_class if LoRA label disagrees with skeleton ground truth
        if gt_type and constraints.get("ifc_class") != gt_type:
            constraints["ifc_class"] = gt_type
            fixes += 1

        # Use skeleton storey (canonical form) for oracle
        if gt_storey:
            constraints["storey_name"] = gt_storey

        # Enrich spatial_relations with subject_type from skeleton
        if spatial_rels:
            subject_type = gt_type or constraints.get("ifc_class", "")
            enriched = []
            for sr in spatial_rels:
                sr = dict(sr)  # copy
                if "subject_type" not in sr:
                    sr["subject_type"] = subject_type
                enriched.append(sr)
            constraints["spatial_relations"] = enriched

        entry = {
            "case_id": case["case_id"],
            "status": "OK",
            "constraints": constraints,
        }
        results.append(entry)

    if fixes:
        print(f"  Fixed {fixes} ifc_class mismatches (LoRA label → skeleton GT)")

    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} precomputed constraints → {OUTPUT}")
    spatial_count = sum(1 for r in results if r["constraints"].get("spatial_relations"))
    print(f"  With spatial_relations: {spatial_count}")
    print(f"  Without spatial_relations: {len(results) - spatial_count}")


if __name__ == "__main__":
    main()
