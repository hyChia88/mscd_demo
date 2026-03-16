#!/usr/bin/env python3
"""
IFC Spatial Relationship Reality Check — AdvancedProject.ifc
Generates statistics + thesis-ready plots for graph enrichment analysis.

Output:
  /tmp/ifc_reality_check/
    ├── stats.txt                    # Full text report
    ├── fig1_entity_types.pdf        # Bar: entity type distribution
    ├── fig2_relationship_types.pdf  # Bar: IFC relationship type counts
    ├── fig3_wall_signatures.pdf     # Bar: wall child signatures (FILLS)
    ├── fig4_storey_breakdown.pdf    # Stacked bar: per-storey element mix
    ├── fig5_next_to_potential.pdf   # Bar: hetero vs homo NEXT_TO
    ├── fig6_oracle_discrimination.pdf  # Grouped bar: 1-hop vs 2-hop oracle Top-1
    └── fig7_enrichment_summary.pdf # Before/after edge counts
"""
import ifcopenshell
import ifcopenshell.util.placement
import ifcopenshell.util.element
from collections import Counter, defaultdict
import os, sys, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
IFC_PATH = "/root/cmu/master_thesis/data_curation/ifc_models/AdvancedProject.ifc"
OUT_DIR = "/tmp/ifc_reality_check"
os.makedirs(OUT_DIR, exist_ok=True)

# Thesis style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
COLORS = {
    "primary": "#2563EB",
    "secondary": "#EA580C",
    "accent": "#16A34A",
    "gray": "#6B7280",
    "light": "#DBEAFE",
    "warn": "#F59E0B",
}
IFC_COLORS = {
    "IfcWallStandardCase": "#4A90D9",
    "IfcWindow": "#F5A623",
    "IfcDoor": "#D0021B",
    "IfcSlab": "#7B8A8B",
    "IfcColumn": "#9B59B6",
    "IfcBeam": "#27AE60",
    "IfcStairFlight": "#E67E22",
    "IfcRailing": "#1ABC9C",
    "IfcStair": "#C0392B",
    "IfcCurtainWall": "#3498DB",
    "IfcPlate": "#95A5A6",
    "IfcMember": "#8E44AD",
    "IfcBuildingElementProxy": "#BDC3C7",
    "IfcWall": "#2C3E50",
    "Other": "#D5D8DC",
}

log_lines = []
def log(s=""):
    print(s)
    log_lines.append(s)

# ═══════════════════════════════════════════════
# LOAD IFC
# ═══════════════════════════════════════════════
log(f"Loading {IFC_PATH} ...")
f = ifcopenshell.open(IFC_PATH)
log(f"Loaded. Schema: {f.schema}")

# ═══════════════════════════════════════════════
# 1. IFC RELATIONSHIP TYPES
# ═══════════════════════════════════════════════
rel_types = Counter()
for inst in f:
    if inst.is_a().startswith("IfcRel"):
        rel_types[inst.is_a()] += 1

log("\n" + "=" * 65)
log("1. IFC RELATIONSHIP TYPES (raw from file)")
log("=" * 65)
total_rels = sum(rel_types.values())
for k, v in rel_types.most_common():
    log(f"  {k:50s} {v:5d}")
log(f"  {'TOTAL':50s} {total_rels:5d}")

# Plot fig2
fig, ax = plt.subplots(figsize=(8, 4.5))
items = rel_types.most_common()
names = [k.replace("IfcRel", "") for k, _ in items]
vals = [v for _, v in items]
bars = ax.barh(range(len(names)), vals, color=COLORS["primary"], edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Count")
ax.set_title("IFC Relationship Types — AdvancedProject.ifc")
for bar, val in zip(bars, vals):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=7, color=COLORS["gray"])
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_relationship_types.pdf")
plt.close()

# ═══════════════════════════════════════════════
# 2. SPATIAL ENTITY TYPES
# ═══════════════════════════════════════════════
ent_types = Counter()
for inst in f:
    if any(inst.is_a(bt) for bt in ["IfcElement", "IfcSpatialStructureElement", "IfcSpace"]):
        if not inst.is_a("IfcOpeningElement"):
            ent_types[inst.is_a()] += 1

log("\n" + "=" * 65)
log("2. SPATIAL ENTITY TYPES")
log("=" * 65)
for k, v in ent_types.most_common():
    log(f"  {k:40s} {v:5d}")
log(f"  {'TOTAL':40s} {sum(ent_types.values()):5d}")

# Plot fig1
fig, ax = plt.subplots(figsize=(7, 4))
items1 = ent_types.most_common()
names1 = [k for k, _ in items1]
vals1 = [v for _, v in items1]
colors1 = [IFC_COLORS.get(n, IFC_COLORS["Other"]) for n in names1]
bars = ax.bar(range(len(names1)), vals1, color=colors1, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(names1)))
ax.set_xticklabels(names1, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Count")
ax.set_title("Spatial Entity Type Distribution")
for bar, val in zip(bars, vals1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(val), ha="center", fontsize=7)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_entity_types.pdf")
plt.close()

# ═══════════════════════════════════════════════
# 3. SPATIAL RELATIONSHIP BREAKDOWN
# ═══════════════════════════════════════════════

# 3a. Containment
log("\n" + "=" * 65)
log("3a. IfcRelContainedInSpatialStructure")
log("=" * 65)
contain = Counter()
storey_elements = defaultdict(list)
for rel in f.by_type("IfcRelContainedInSpatialStructure"):
    s = rel.RelatingStructure
    sname = s.Name or "unnamed"
    for el in rel.RelatedElements:
        if not el.is_a("IfcOpeningElement"):
            contain[(s.is_a(), sname, el.is_a())] += 1
            if s.is_a("IfcBuildingStorey"):
                storey_elements[sname].append(el)
for (st, sn, et), c in contain.most_common(30):
    log(f"  {st}:{sn} → {et}: {c}")

# 3b. FILLS chain
log("\n" + "=" * 65)
log("3b. IfcRelFillsElement + IfcRelVoidsElement (FILLS chain)")
log("=" * 65)
opening_to_host = {}
for rel in f.by_type("IfcRelVoidsElement"):
    opening_to_host[rel.RelatedOpeningElement.GlobalId] = rel.RelatingBuildingElement

fills_edges = []
fills_types = Counter()
for rel in f.by_type("IfcRelFillsElement"):
    filling = rel.RelatedBuildingElement
    opening = rel.RelatingOpeningElement
    host = opening_to_host.get(opening.GlobalId)
    fills_types[filling.is_a()] += 1
    if host:
        fills_edges.append((filling, host))

log(f"  Total IfcRelFillsElement: {len(f.by_type('IfcRelFillsElement'))}")
log(f"  Resolved FILLS edges (filler→host): {len(fills_edges)}")
for k, v in fills_types.most_common():
    log(f"    {k}: {v}")

# Wall children analysis
wall_children = defaultdict(list)
for filler, host in fills_edges:
    wall_children[host.GlobalId].append(filler)

log(f"\n  Unique host walls: {len(wall_children)}")
child_sigs = Counter()
for wguid, children in wall_children.items():
    sig = tuple(sorted(c.is_a() for c in children))
    child_sigs[sig] += 1
log("  Wall child signatures (FILLS children):")
for sig, cnt in child_sigs.most_common(20):
    label = ", ".join(sig)
    log(f"    [{cnt}x] {label}")

# Plot fig3: Wall signatures
fig, ax = plt.subplots(figsize=(7, 4))
sig_items = child_sigs.most_common(15)
sig_labels = []
for sig, cnt in sig_items:
    short = []
    type_c = Counter(sig)
    for t, c in type_c.most_common():
        short.append(f"{c}×{t.replace('Ifc','')}")
    sig_labels.append(" + ".join(short))
sig_vals = [c for _, c in sig_items]
bars = ax.barh(range(len(sig_labels)), sig_vals, color=COLORS["primary"],
               edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(sig_labels)))
ax.set_yticklabels(sig_labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Number of Walls")
ax.set_title("Wall FILLS-Children Signatures")
for bar, val in zip(bars, sig_vals):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=8)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_wall_signatures.pdf")
plt.close()

# 3c. Wall-Wall connections
log("\n" + "=" * 65)
log("3c. IfcRelConnectsPathElements (wall-wall topology)")
log("=" * 65)
cpp = f.by_type("IfcRelConnectsPathElements")
log(f"  Total: {len(cpp)}")
conn_types = Counter()
for rel in cpp:
    t1 = rel.RelatingElement.is_a() if rel.RelatingElement else "None"
    t2 = rel.RelatedElement.is_a() if rel.RelatedElement else "None"
    conn_types[(t1, t2)] += 1
for (t1, t2), c in conn_types.most_common():
    log(f"  {t1} -- {t2}: {c}")

# 3d. Space boundaries
log("\n" + "=" * 65)
log("3d. IfcRelSpaceBoundary")
log("=" * 65)
sb = f.by_type("IfcRelSpaceBoundary")
log(f"  Total: {len(sb)}")
sb_types = Counter()
for rel in sb:
    el = rel.RelatedBuildingElement
    if el:
        sb_types[el.is_a()] += 1
for k, v in sb_types.most_common():
    log(f"  Bounded by {k}: {v}")

# 3e. Aggregates
log("\n" + "=" * 65)
log("3e. IfcRelAggregates")
log("=" * 65)
agg_types = Counter()
for rel in f.by_type("IfcRelAggregates"):
    parent = rel.RelatingObject
    for child in rel.RelatedObjects:
        agg_types[(parent.is_a(), child.is_a())] += 1
for (pt, ct), c in agg_types.most_common():
    log(f"  {pt} → {ct}: {c}")

# 3f. Generic connects
log("\n" + "=" * 65)
log("3f. IfcRelConnectsElements (generic, excl. path)")
log("=" * 65)
ce = f.by_type("IfcRelConnectsElements")
pure = [r for r in ce if r.is_a() == "IfcRelConnectsElements"]
log(f"  Pure IfcRelConnectsElements: {len(pure)}")

# 3g. Ports
log("\n" + "=" * 65)
log("3g. Port Connections")
log("=" * 65)
log(f"  IfcRelConnectsPortToElement: {len(f.by_type('IfcRelConnectsPortToElement'))}")
log(f"  IfcRelConnectsPorts: {len(f.by_type('IfcRelConnectsPorts'))}")

# 3h. Materials
log("\n" + "=" * 65)
log("3h. IfcRelAssociatesMaterial")
log("=" * 65)
mat_rels = f.by_type("IfcRelAssociatesMaterial")
log(f"  Total: {len(mat_rels)}")

# ═══════════════════════════════════════════════
# 4. PER-STOREY BREAKDOWN
# ═══════════════════════════════════════════════
log("\n" + "=" * 65)
log("4. PER-STOREY ELEMENT + EDGE COUNTS")
log("=" * 65)

storey_order = sorted(storey_elements.keys(),
                      key=lambda s: (int(m.group(1)) if (m := __import__('re').search(r'(-?\d+)', s)) else 99, s))

storey_type_matrix = {}  # {storey: {ifc_type: count}}
key_types = ["IfcWallStandardCase", "IfcWindow", "IfcDoor", "IfcSlab",
             "IfcColumn", "IfcBeam", "IfcRailing", "IfcStairFlight"]

for sname in storey_order:
    elems = storey_elements[sname]
    type_counts = Counter(e.is_a() for e in elems)
    storey_guids = {e.GlobalId for e in elems}
    fills_on_storey = sum(1 for filler, host in fills_edges
                          if filler.GlobalId in storey_guids)
    log(f"\n  {sname} ({len(elems)} elements, {fills_on_storey} FILLS)")
    storey_type_matrix[sname] = type_counts
    for t, c in type_counts.most_common():
        log(f"    {t}: {c}")

# Plot fig4: Stacked bar per storey
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(storey_order))
bottom = np.zeros(len(storey_order))
for ifc_type in key_types:
    vals = [storey_type_matrix.get(s, {}).get(ifc_type, 0) for s in storey_order]
    if sum(vals) == 0:
        continue
    color = IFC_COLORS.get(ifc_type, IFC_COLORS["Other"])
    ax.bar(x, vals, bottom=bottom, label=ifc_type.replace("Ifc", ""),
           color=color, edgecolor="white", linewidth=0.3)
    bottom += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels([s[:20] for s in storey_order], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Element Count")
ax.set_title("Per-Storey Element Distribution")
ax.legend(loc="upper right", fontsize=7, ncol=2)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_storey_breakdown.pdf")
plt.close()

# ═══════════════════════════════════════════════
# 5. NEXT_TO ENRICHMENT POTENTIAL
# ═══════════════════════════════════════════════
log("\n" + "=" * 65)
log("5. NEXT_TO ENRICHMENT POTENTIAL")
log("=" * 65)

def get_centroid(elem):
    try:
        m = ifcopenshell.util.placement.get_local_placement(elem.ObjectPlacement)
        return (m[0][3], m[1][3], m[2][3])
    except:
        return None

next_to_edges = 0
hetero_next_to = 0
homo_next_to = 0

wall_sequences = []
wall_seq_details = []  # for oracle analysis
for wguid, children in wall_children.items():
    if len(children) < 2:
        continue
    child_pos = []
    for c in children:
        pos = get_centroid(c)
        if pos:
            child_pos.append((c, pos))
    if len(child_pos) < 2:
        continue

    xs = [p[1][0] for p in child_pos]
    ys = [p[1][1] for p in child_pos]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    sort_axis = 0 if x_range >= y_range else 1
    child_pos.sort(key=lambda cp: cp[1][sort_axis])

    seq = [(c.is_a(), c.GlobalId) for c, _ in child_pos]
    wall_sequences.append(seq)
    wall_seq_details.append((wguid, child_pos))

    for i in range(len(seq) - 1):
        next_to_edges += 1
        if seq[i][0] != seq[i + 1][0]:
            hetero_next_to += 1
        else:
            homo_next_to += 1

log(f"  Walls with ≥2 children: {len(wall_sequences)}")
log(f"  Total NEXT_TO edges (ordered pairs): {next_to_edges}")
log(f"    Heterogeneous (cross-type): {hetero_next_to}")
log(f"    Homogeneous (same-type): {homo_next_to}")
log(f"    Hetero ratio: {hetero_next_to / max(next_to_edges, 1):.1%}")

log(f"\n  Wall sequences (type patterns):")
seq_patterns = Counter()
for seq in wall_sequences:
    pattern = " → ".join(t.replace("Ifc", "") for t, _ in seq)
    seq_patterns[pattern] += 1
for pat, cnt in seq_patterns.most_common(20):
    log(f"    [{cnt}x] {pat}")

# Plot fig5: NEXT_TO potential
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# Left: hetero vs homo
ax = axes[0]
ax.bar(["Heterogeneous\n(cross-type)", "Homogeneous\n(same-type)"],
       [hetero_next_to, homo_next_to],
       color=[COLORS["accent"], COLORS["gray"]], edgecolor="white")
ax.set_ylabel("NEXT_TO Edge Count")
ax.set_title("NEXT_TO Edge Types")
for i, v in enumerate([hetero_next_to, homo_next_to]):
    ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)

# Right: wall sequence length distribution
ax = axes[1]
seq_lens = [len(seq) for seq in wall_sequences]
len_counts = Counter(seq_lens)
lens_sorted = sorted(len_counts.keys())
ax.bar(lens_sorted, [len_counts[l] for l in lens_sorted],
       color=COLORS["primary"], edgecolor="white")
ax.set_xlabel("Children per Wall")
ax.set_ylabel("Number of Walls")
ax.set_title("FILLS Children per Host Wall")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_next_to_potential.pdf")
plt.close()

# ═══════════════════════════════════════════════
# 6. EDGE SUMMARY: BEFORE vs AFTER ENRICHMENT
# ═══════════════════════════════════════════════
log("\n" + "=" * 65)
log("6. GRAPH EDGE SUMMARY — BEFORE vs AFTER ENRICHMENT")
log("=" * 65)

# CONTINUOUS count
cont_count = 0
for wall in f.by_type("IfcWallStandardCase"):
    psets = ifcopenshell.util.element.get_psets(wall)
    for pset_name, props in psets.items():
        if "constraint" in pset_name.lower():
            if "Top Constraint" in props:
                tc = props["Top Constraint"]
                if tc and tc != "Unconnected":
                    cont_count += 1
                    break

# ADJACENT_TO: approximate from geometry (elements on same storey within 1500mm)
# We'll use a simpler count based on what's in Neo4j already
adj_count_approx = 200  # from prior Neo4j analysis

edge_before = {
    "FILLS": len(fills_edges),
    "CONTINUOUS": cont_count,
    "ADJACENT_TO": adj_count_approx,
    "CONTAINS": sum(contain.values()),
}
edge_after = dict(edge_before)
edge_after["NEXT_TO"] = next_to_edges
edge_after["WALL_CONNECTS"] = len(cpp)

log(f"\n  {'Edge Type':<20s} {'Before':>8s} {'After':>8s} {'Delta':>8s}")
log(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
all_edge_types = sorted(set(list(edge_before.keys()) + list(edge_after.keys())))
for et in all_edge_types:
    b = edge_before.get(et, 0)
    a = edge_after.get(et, 0)
    d = a - b
    log(f"  {et:<20s} {b:>8d} {a:>8d} {'+' + str(d) if d > 0 else str(d):>8s}")
total_b = sum(edge_before.values())
total_a = sum(edge_after.values())
log(f"  {'TOTAL':<20s} {total_b:>8d} {total_a:>8d} {'+' + str(total_a - total_b):>8s}")

# Plot fig7: Before/after comparison
fig, ax = plt.subplots(figsize=(7, 4))
edge_types_plot = ["FILLS", "CONTINUOUS", "ADJACENT_TO", "NEXT_TO", "WALL_CONNECTS"]
before_vals = [edge_before.get(e, 0) for e in edge_types_plot]
after_vals = [edge_after.get(e, 0) for e in edge_types_plot]
x = np.arange(len(edge_types_plot))
w = 0.35
ax.bar(x - w / 2, before_vals, w, label="Current Graph", color=COLORS["gray"], edgecolor="white")
ax.bar(x + w / 2, after_vals, w, label="After Enrichment", color=COLORS["primary"], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(edge_types_plot, fontsize=9)
ax.set_ylabel("Edge Count")
ax.set_title("Graph Edges — Before vs After Enrichment")
ax.legend()
for i, (b, a) in enumerate(zip(before_vals, after_vals)):
    if a > b:
        ax.text(i + w / 2, a + 3, f"+{a - b}", ha="center", fontsize=7, color=COLORS["primary"])
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig7_enrichment_summary.pdf")
plt.close()

# ═══════════════════════════════════════════════
# 7. ORACLE — THEORETICAL MAX DISCRIMINATION
# ═══════════════════════════════════════════════
log("\n" + "=" * 65)
log("7. ORACLE — THEORETICAL MAX DISCRIMINATION")
log("=" * 65)

# 1-hop fingerprints: FILLS only (current state)
elem_fp_current = defaultdict(set)
for filler, host in fills_edges:
    fg = filler.GlobalId
    hg = host.GlobalId
    elem_fp_current[fg].add(("FILLS", host.is_a()))
    elem_fp_current[hg].add(("FILLED_BY", filler.is_a()))

# 1-hop + NEXT_TO
elem_fp_1hop = defaultdict(set)
for filler, host in fills_edges:
    fg = filler.GlobalId
    hg = host.GlobalId
    elem_fp_1hop[fg].add(("FILLS", host.is_a()))
    elem_fp_1hop[hg].add(("FILLED_BY", filler.is_a()))

for seq in wall_sequences:
    for i in range(len(seq) - 1):
        g1, g2 = seq[i][1], seq[i + 1][1]
        t1, t2 = seq[i][0], seq[i + 1][0]
        elem_fp_1hop[g1].add(("NEXT_TO", t2))
        elem_fp_1hop[g2].add(("NEXT_TO", t1))

# 2-hop: FILLS + wall siblings (count) + NEXT_TO + position ordinal
elem_fp_2hop = {}
for filler, host in fills_edges:
    fg = filler.GlobalId
    hg = host.GlobalId
    fp = set()
    fp.add(("FILLS", host.is_a()))
    siblings = wall_children.get(hg, [])
    n_siblings = len(siblings) - 1  # exclude self
    sib_types = Counter(s.is_a() for s in siblings if s.GlobalId != fg)
    for st, sc in sib_types.items():
        fp.add(("WALL_SIBLING", st, sc))
    fp.add(("SIBLING_COUNT", n_siblings))
    fp.update(elem_fp_1hop.get(fg, set()) - {("FILLS", host.is_a())})
    elem_fp_2hop[fg] = fp

# 2-hop + position ordinal (rank on wall)
elem_fp_2hop_pos = {}
for wguid, child_pos_list in wall_seq_details:
    for rank, (elem, pos) in enumerate(child_pos_list):
        fg = elem.GlobalId
        fp = set(elem_fp_2hop.get(fg, set()))
        fp.add(("POSITION", rank, len(child_pos_list)))
        elem_fp_2hop_pos[fg] = fp
# Also include elements with only 1 child (no NEXT_TO but still have FILLS)
for fg, fp_set in elem_fp_2hop.items():
    if fg not in elem_fp_2hop_pos:
        elem_fp_2hop_pos[fg] = fp_set

# Compute oracle stats per storey × ifc_type
oracle_results = []  # list of dicts for plotting

target_types = ["IfcWindow", "IfcDoor", "IfcWallStandardCase"]

log("\n  Current (FILLS-only 1-hop):")
for sname in storey_order:
    elems = storey_elements[sname]
    by_type = defaultdict(list)
    for e in elems:
        by_type[e.is_a()].append(e)

    for ifc_type in target_types:
        type_elems = by_type.get(ifc_type, [])
        if not type_elems:
            continue
        fps = Counter()
        for e in type_elems:
            fp = frozenset(elem_fp_current.get(e.GlobalId, set()))
            fps[fp] += 1
        unique = len(fps)
        total = len(type_elems)
        max_pool = max(fps.values()) if fps else 0
        avg_top1 = sum(1.0 / cnt for fp, cnt in fps.items() for _ in range(cnt)) / total if total else 0
        oracle_results.append({
            "storey": sname, "ifc_type": ifc_type, "stage": "current",
            "total": total, "unique_fps": unique, "max_dup": max_pool,
            "oracle_top1": avg_top1
        })
        log(f"    {sname:25s} | {ifc_type:25s} n={total:3d}  "
            f"unique={unique:3d}  max_dup={max_pool:3d}  "
            f"oracle_top1={avg_top1:.1%}")

log("\n  +NEXT_TO (1-hop enriched):")
for sname in storey_order:
    elems = storey_elements[sname]
    by_type = defaultdict(list)
    for e in elems:
        by_type[e.is_a()].append(e)

    for ifc_type in target_types:
        type_elems = by_type.get(ifc_type, [])
        if not type_elems:
            continue
        fps = Counter()
        for e in type_elems:
            fp = frozenset(elem_fp_1hop.get(e.GlobalId, set()))
            fps[fp] += 1
        unique = len(fps)
        total = len(type_elems)
        max_pool = max(fps.values()) if fps else 0
        avg_top1 = sum(1.0 / cnt for fp, cnt in fps.items() for _ in range(cnt)) / total if total else 0
        oracle_results.append({
            "storey": sname, "ifc_type": ifc_type, "stage": "+NEXT_TO",
            "total": total, "unique_fps": unique, "max_dup": max_pool,
            "oracle_top1": avg_top1
        })
        log(f"    {sname:25s} | {ifc_type:25s} n={total:3d}  "
            f"unique={unique:3d}  max_dup={max_pool:3d}  "
            f"oracle_top1={avg_top1:.1%}")

log("\n  +2-hop (siblings + count):")
for sname in storey_order:
    elems = storey_elements[sname]
    by_type = defaultdict(list)
    for e in elems:
        by_type[e.is_a()].append(e)

    for ifc_type in target_types:
        type_elems = by_type.get(ifc_type, [])
        if not type_elems:
            continue
        fps = Counter()
        for e in type_elems:
            fp = frozenset(elem_fp_2hop.get(e.GlobalId, frozenset()))
            fps[fp] += 1
        unique = len(fps)
        total = len(type_elems)
        max_pool = max(fps.values()) if fps else 0
        avg_top1 = sum(1.0 / cnt for fp, cnt in fps.items() for _ in range(cnt)) / total if total else 0
        oracle_results.append({
            "storey": sname, "ifc_type": ifc_type, "stage": "+2-hop",
            "total": total, "unique_fps": unique, "max_dup": max_pool,
            "oracle_top1": avg_top1
        })
        log(f"    {sname:25s} | {ifc_type:25s} n={total:3d}  "
            f"unique={unique:3d}  max_dup={max_pool:3d}  "
            f"oracle_top1={avg_top1:.1%}")

log("\n  +position ordinal (2-hop + rank on wall):")
for sname in storey_order:
    elems = storey_elements[sname]
    by_type = defaultdict(list)
    for e in elems:
        by_type[e.is_a()].append(e)

    for ifc_type in target_types:
        type_elems = by_type.get(ifc_type, [])
        if not type_elems:
            continue
        fps = Counter()
        for e in type_elems:
            fp = frozenset(elem_fp_2hop_pos.get(e.GlobalId, frozenset()))
            fps[fp] += 1
        unique = len(fps)
        total = len(type_elems)
        max_pool = max(fps.values()) if fps else 0
        avg_top1 = sum(1.0 / cnt for fp, cnt in fps.items() for _ in range(cnt)) / total if total else 0
        oracle_results.append({
            "storey": sname, "ifc_type": ifc_type, "stage": "+position",
            "total": total, "unique_fps": unique, "max_dup": max_pool,
            "oracle_top1": avg_top1
        })
        log(f"    {sname:25s} | {ifc_type:25s} n={total:3d}  "
            f"unique={unique:3d}  max_dup={max_pool:3d}  "
            f"oracle_top1={avg_top1:.1%}")

# ═══════════════════════════════════════════════
# Plot fig6: Oracle discrimination comparison
# ═══════════════════════════════════════════════
# Aggregate across storeys per ifc_type per stage
stages = ["current", "+NEXT_TO", "+2-hop", "+position"]
stage_colors = [COLORS["gray"], COLORS["warn"], COLORS["primary"], COLORS["accent"]]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
for ax, ifc_type in zip(axes, target_types):
    # Aggregate: weighted average oracle_top1 across storeys
    stage_vals = []
    for stage in stages:
        matching = [r for r in oracle_results if r["ifc_type"] == ifc_type and r["stage"] == stage]
        if matching:
            total_n = sum(r["total"] for r in matching)
            weighted = sum(r["oracle_top1"] * r["total"] for r in matching) / max(total_n, 1)
            stage_vals.append(weighted)
        else:
            stage_vals.append(0)

    x_pos = np.arange(len(stages))
    bars = ax.bar(x_pos, [v * 100 for v in stage_vals], color=stage_colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=8)
    ax.set_title(ifc_type.replace("Ifc", ""), fontsize=11)
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, stage_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0%}", ha="center", fontsize=8)

axes[0].set_ylabel("Oracle Top-1 Accuracy (%)")
fig.suptitle("Theoretical Maximum Top-1 by Graph Enrichment Stage", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig6_oracle_discrimination.pdf")
plt.close()

# ═══════════════════════════════════════════════
# 8. SUMMARY TABLE
# ═══════════════════════════════════════════════
log("\n" + "=" * 65)
log("8. SUMMARY — GRAPH ENRICHMENT ROADMAP")
log("=" * 65)
log(f"""
  Current graph: {len(fills_edges)} FILLS + {cont_count} CONTINUOUS + ~{adj_count_approx} ADJACENT_TO
  After enrichment: +{next_to_edges} NEXT_TO + {len(cpp)} WALL_CONNECTS

  Key findings:
  - {len(wall_children)} unique host walls hold {len(fills_edges)} fillers
  - {len(wall_sequences)} walls have ≥2 children → {next_to_edges} NEXT_TO edges
  - Heterogeneous NEXT_TO (cross-type): {hetero_next_to}/{next_to_edges} ({hetero_next_to/max(next_to_edges,1):.0%})
  - Wall-wall path connections: {len(cpp)} (from Revit topology)
  - Space boundaries: {len(sb)} (rich but underutilized)

  Implication for LoRA_5:
  - Homogeneous NEXT_TO ({homo_next_to} edges) = low discrimination value
  - Heterogeneous NEXT_TO ({hetero_next_to} edges) = HIGH discrimination value
  - Position ordinal adds unique fingerprint even for homo sequences
""")

# Save stats
with open(f"{OUT_DIR}/stats.txt", "w") as fout:
    fout.write("\n".join(log_lines))

# Save oracle results as JSONL
with open(f"{OUT_DIR}/oracle_results.jsonl", "w") as fout:
    for r in oracle_results:
        fout.write(json.dumps(r) + "\n")

log(f"\nAll outputs saved to {OUT_DIR}/")
log("Done.")
