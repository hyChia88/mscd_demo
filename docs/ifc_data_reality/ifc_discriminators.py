#!/usr/bin/env python3
"""Explore ALL possible discriminative features in AdvancedProject.ifc beyond topology."""
import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement
from collections import Counter, defaultdict
import json

IFC_PATH = "/root/cmu/master_thesis/data_curation/ifc_models/AdvancedProject.ifc"
f = ifcopenshell.open(IFC_PATH)

# ═══════════════════════════════════════════════
# 1. SPACE NAMES — do elements have space assignments?
# ═══════════════════════════════════════════════
print("=" * 65)
print("1. IfcSpace INVENTORY")
print("=" * 65)
spaces = f.by_type("IfcSpace")
print(f"  Total IfcSpace: {len(spaces)}")
for sp in spaces:
    name = sp.Name or "—"
    long_name = sp.LongName or "—"
    # Get containing storey
    container = ifcopenshell.util.element.get_container(sp)
    storey = container.Name if container else "—"
    # Get elements in this space
    decomp = ifcopenshell.util.element.get_decomposition(sp)
    elem_types = Counter(e.is_a() for e in decomp if not e.is_a("IfcSpace"))
    print(f"  {name:20s} LongName={long_name:30s} Storey={storey:25s} Children={dict(elem_types)}")

# Check if elements reference spaces via IfcRelContainedInSpatialStructure
print(f"\n  Elements contained directly in IfcSpace:")
space_contained = 0
for rel in f.by_type("IfcRelContainedInSpatialStructure"):
    if rel.RelatingStructure.is_a("IfcSpace"):
        for el in rel.RelatedElements:
            space_contained += 1
            print(f"    {el.is_a()} '{el.Name}' in space '{rel.RelatingStructure.Name}'")
print(f"  Total: {space_contained}")

# ═══════════════════════════════════════════════
# 2. PROPERTY SETS — what psets do windows/doors have?
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("2. PROPERTY SETS on Windows (sample 5)")
print("=" * 65)
windows = f.by_type("IfcWindow")
for w in windows[:5]:
    psets = ifcopenshell.util.element.get_psets(w)
    print(f"\n  Window '{w.Name}' GUID={w.GlobalId[:8]}...")
    for pset_name, props in psets.items():
        print(f"    [{pset_name}]")
        for k, v in props.items():
            if k != "id":
                print(f"      {k}: {v}")

print("\n" + "=" * 65)
print("3. PROPERTY SETS on Doors (sample 5)")
print("=" * 65)
doors = f.by_type("IfcDoor")
for d in doors[:5]:
    psets = ifcopenshell.util.element.get_psets(d)
    print(f"\n  Door '{d.Name}' GUID={d.GlobalId[:8]}...")
    for pset_name, props in psets.items():
        print(f"    [{pset_name}]")
        for k, v in props.items():
            if k != "id":
                print(f"      {k}: {v}")

# ═══════════════════════════════════════════════
# 3. UNIQUE PROPERTY VALUES — what actually varies?
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("4. DISCRIMINATIVE PROPERTY ANALYSIS — IfcWindow")
print("=" * 65)

# Collect all property values across all windows
win_props = defaultdict(Counter)  # {prop_name: Counter({value: count})}
win_names = Counter()
win_types = Counter()
win_descriptions = Counter()
win_object_types = Counter()
win_tag = Counter()

for w in windows:
    win_names[w.Name or "—"] += 1
    win_descriptions[w.Description or "—"] += 1
    win_object_types[getattr(w, "ObjectType", None) or "—"] += 1
    win_tag[getattr(w, "Tag", None) or "—"] += 1
    psets = ifcopenshell.util.element.get_psets(w)
    for pset_name, props in psets.items():
        for k, v in props.items():
            if k != "id":
                win_props[f"{pset_name}.{k}"][str(v)] += 1

print(f"\n  Attribute: Name — {len(win_names)} unique values")
for v, c in win_names.most_common(10):
    print(f"    '{v}': {c}")

print(f"\n  Attribute: Description — {len(win_descriptions)} unique values")
for v, c in win_descriptions.most_common(10):
    print(f"    '{v}': {c}")

print(f"\n  Attribute: ObjectType — {len(win_object_types)} unique values")
for v, c in win_object_types.most_common(10):
    print(f"    '{v}': {c}")

print(f"\n  Attribute: Tag — {len(win_tag)} unique values")
for v, c in win_tag.most_common(10):
    print(f"    '{v}': {c}")

# Find properties with HIGH variance (many unique values)
print(f"\n  Properties ranked by uniqueness (out of {len(windows)} windows):")
prop_uniqueness = []
for prop_name, value_counts in win_props.items():
    n_unique = len(value_counts)
    max_dup = max(value_counts.values())
    prop_uniqueness.append((prop_name, n_unique, max_dup))
prop_uniqueness.sort(key=lambda x: -x[1])
for prop_name, n_unique, max_dup in prop_uniqueness[:30]:
    entropy_pct = n_unique / len(windows) * 100
    print(f"    {prop_name:50s} unique={n_unique:4d}  max_dup={max_dup:4d}  entropy={entropy_pct:.1f}%")

# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("5. DISCRIMINATIVE PROPERTY ANALYSIS — IfcDoor")
print("=" * 65)

door_props = defaultdict(Counter)
door_names = Counter()
door_types = Counter()

for d in doors:
    door_names[d.Name or "—"] += 1
    door_types[getattr(d, "ObjectType", None) or "—"] += 1
    psets = ifcopenshell.util.element.get_psets(d)
    for pset_name, props in psets.items():
        for k, v in props.items():
            if k != "id":
                door_props[f"{pset_name}.{k}"][str(v)] += 1

print(f"\n  Attribute: Name — {len(door_names)} unique values")
for v, c in door_names.most_common(10):
    print(f"    '{v}': {c}")

print(f"\n  Properties ranked by uniqueness (out of {len(doors)} doors):")
door_uniqueness = []
for prop_name, value_counts in door_props.items():
    n_unique = len(value_counts)
    max_dup = max(value_counts.values())
    door_uniqueness.append((prop_name, n_unique, max_dup))
door_uniqueness.sort(key=lambda x: -x[1])
for prop_name, n_unique, max_dup in door_uniqueness[:30]:
    entropy_pct = n_unique / len(doors) * 100
    print(f"    {prop_name:50s} unique={n_unique:4d}  max_dup={max_dup:4d}  entropy={entropy_pct:.1f}%")

# ═══════════════════════════════════════════════
# 4. WALL PROPERTIES — what varies across walls?
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("6. DISCRIMINATIVE PROPERTY ANALYSIS — IfcWallStandardCase")
print("=" * 65)
walls = f.by_type("IfcWallStandardCase")

wall_props = defaultdict(Counter)
wall_names = Counter()

for w in walls:
    wall_names[w.Name or "—"] += 1
    psets = ifcopenshell.util.element.get_psets(w)
    for pset_name, props in psets.items():
        for k, v in props.items():
            if k != "id":
                wall_props[f"{pset_name}.{k}"][str(v)] += 1

print(f"\n  Attribute: Name — {len(wall_names)} unique values")
for v, c in wall_names.most_common(10):
    print(f"    '{v}': {c}")

print(f"\n  Properties ranked by uniqueness (out of {len(walls)} walls):")
wall_uniqueness = []
for prop_name, value_counts in wall_props.items():
    n_unique = len(value_counts)
    max_dup = max(value_counts.values())
    wall_uniqueness.append((prop_name, n_unique, max_dup))
wall_uniqueness.sort(key=lambda x: -x[1])
for prop_name, n_unique, max_dup in wall_uniqueness[:30]:
    entropy_pct = n_unique / len(walls) * 100
    print(f"    {prop_name:50s} unique={n_unique:4d}  max_dup={max_dup:4d}  entropy={entropy_pct:.1f}%")

# ═══════════════════════════════════════════════
# 5. MATERIAL — does material discriminate?
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("7. MATERIAL DISCRIMINATION")
print("=" * 65)

def get_material(elem):
    """Extract material name(s) from element."""
    mats = []
    for rel in f.by_type("IfcRelAssociatesMaterial"):
        if elem in rel.RelatedObjects:
            mat = rel.RelatingMaterial
            if mat.is_a("IfcMaterial"):
                mats.append(mat.Name)
            elif mat.is_a("IfcMaterialLayerSetUsage"):
                ls = mat.ForLayerSet
                for layer in ls.MaterialLayers:
                    if layer.Material:
                        mats.append(layer.Material.Name)
            elif mat.is_a("IfcMaterialLayerSet"):
                for layer in mat.MaterialLayers:
                    if layer.Material:
                        mats.append(layer.Material.Name)
            elif mat.is_a("IfcMaterialList"):
                for m in mat.Materials:
                    mats.append(m.Name)
    return tuple(sorted(set(mats))) if mats else ("—",)

# Sample materials for key types
for ifc_type, label in [("IfcWindow", "Window"), ("IfcDoor", "Door"), ("IfcWallStandardCase", "Wall")]:
    elems = f.by_type(ifc_type)
    mat_counter = Counter()
    for e in elems[:50]:  # sample first 50
        mat_counter[get_material(e)] += 1
    print(f"\n  {label} materials (sample {min(50, len(elems))}/{len(elems)}):")
    for mat, cnt in mat_counter.most_common():
        print(f"    {mat}: {cnt}")

# ═══════════════════════════════════════════════
# 6. DIMENSIONS — do width/height vary?
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("8. DIMENSION DISCRIMINATION (OverallWidth × OverallHeight)")
print("=" * 65)

for ifc_type, label in [("IfcWindow", "Window"), ("IfcDoor", "Door")]:
    elems = f.by_type(ifc_type)
    dim_counter = Counter()
    for e in elems:
        w = getattr(e, "OverallWidth", None)
        h = getattr(e, "OverallHeight", None)
        dim_counter[(w, h)] += 1
    print(f"\n  {label} dimensions ({len(dim_counter)} unique combos / {len(elems)} total):")
    for (w, h), cnt in dim_counter.most_common(15):
        print(f"    {w}mm × {h}mm : {cnt}")

# ═══════════════════════════════════════════════
# 7. TYPE OBJECT — IfcWindowType / IfcDoorType
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("9. TYPE OBJECTS (IfcWindowStyle / IfcDoorStyle)")
print("=" * 65)

for ifc_type, label in [("IfcWindow", "Window"), ("IfcDoor", "Door")]:
    elems = f.by_type(ifc_type)
    type_counter = Counter()
    for e in elems:
        etype = ifcopenshell.util.element.get_type(e)
        if etype:
            type_counter[etype.Name or "unnamed"] += 1
        else:
            type_counter["NO_TYPE"] += 1
    print(f"\n  {label} types ({len(type_counter)} unique / {len(elems)} total):")
    for tname, cnt in type_counter.most_common():
        print(f"    '{tname}': {cnt}")

# ═══════════════════════════════════════════════
# 8. SUMMARY: per-storey discrimination with ALL features
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("10. COMBINED FINGERPRINT (type + dim + material + name + storey)")
print("=" * 65)

storey_elements = defaultdict(list)
for rel in f.by_type("IfcRelContainedInSpatialStructure"):
    s = rel.RelatingStructure
    if s.is_a("IfcBuildingStorey"):
        for el in rel.RelatedElements:
            if not el.is_a("IfcOpeningElement"):
                storey_elements[s.Name].append(el)

for ifc_type in ["IfcWindow", "IfcDoor"]:
    print(f"\n  --- {ifc_type} ---")
    for sname in sorted(storey_elements.keys()):
        elems = [e for e in storey_elements[sname] if e.is_a(ifc_type)]
        if not elems:
            continue
        fp_counter = Counter()
        for e in elems:
            w = getattr(e, "OverallWidth", None)
            h = getattr(e, "OverallHeight", None)
            etype = ifcopenshell.util.element.get_type(e)
            tname = etype.Name if etype else "—"
            name = e.Name or "—"
            fp = (name, tname, w, h)
            fp_counter[fp] += 1
        unique = len(fp_counter)
        max_dup = max(fp_counter.values())
        avg_top1 = sum(1.0/c for fp, c in fp_counter.items() for _ in range(c)) / len(elems)
        print(f"    {sname:25s} n={len(elems):3d}  unique_attr_fps={unique:3d}  "
              f"max_dup={max_dup:3d}  attr_oracle_top1={avg_top1:.1%}")
        for fp, cnt in fp_counter.most_common(5):
            print(f"      {fp}: {cnt}")

print("\nDone.")
