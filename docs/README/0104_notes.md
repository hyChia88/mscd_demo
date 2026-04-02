# 0104_notes:

## 1. Dataset Optimization
1. Fix blender camera
2. Regenerate and add data augmentation
see (mscd_demo/output/lora6_v2_ap_20260331/topology_analysis, mscd_demo/output/lora6_v2_ap_20260331/topology_analysis/ap_held_out/representative_family_gallery.png)
![image](../../output/lora6_v2_ap_20260331/topology_analysis/lora6_ap_all/augmentation_flow_alluvial.png) 
ground truth:
```
{
  "case_id": "AP_SK_078",
  "bench": {
    "group": "C",
    "condition": "C1"
  },
  "difficulty_tags": {
    "tier": "Tier 3",
    "requires_relation": true,
    "spatial_predicate": "FILLS",
    "pattern": "FILLS_RELATION"
  },
  "ground_truth": {
    "target_guid": "0ducQKkW5EGQM9rQ5zdPvI",
    "target_storey": "-1 - Garage",
    "target_ifc_class": "IfcDoor",
    "target_name": "M_Single-Flush:0762 x 2032mm:819560"
  },
  "inputs": {
    "chat_history": [
      {
        "role": "Inspector",
        "text": "The door in the wall opening on Garage needs QA review."
      }
    ],
    "chat_quality": "clear",
    "project_context": {
      "4d_task_status": "AP held-out evaluation",
      "project_phase": "LoRA6-v2 AP benchmark"
    },
    "images": [
      "datasets/synth_v0.5_ap/imgs/AP_SK_078_site.png"
    ],
    "floorplan_patch": "datasets/synth_v0.5_ap/floorplans/AP_SK_078_floorplan.png"
  },
  "labels": {
    "constraints": {
      "storey_name": "-1",
      "ifc_class": "IfcDoor",
      "space_name": null,
      "spatial_relations": [
        {
          "predicate": "FILLS",
          "object_type": "IfcWall",
          "confidence": 1.0,
          "object_material": "Interior Wall A",
          "host_name": "Basic Wall:MockUp Storage Wall"
        },
        {
          "predicate": "NEXT_TO",
          "object_type": "IfcDoor",
          "direction": "left",
          "confidence": 1.0
        }
      ],
      "position_context": "2nd of 2 openings on the same wall"
    }
  },
  "query_text": "The door in the wall opening on Garage needs QA review."
}
```

raw output:
```
{
  "case_id": "AP_SK_078",
  "condition": "AP_EVAL",
  "constraints": {
    "storey_name": "-1",
    "ifc_class": "IfcDoor",
    "near_keywords": [],
    "relations": [],
    "space_name": null,
    "target_name_keyword": null,
    "neighbor_type": null,
    "spatial_relations": [
      {
        "predicate": "FILLS",
        "object_type": "IfcWall",
        "confidence": 1.0,
        "object_material": "Interior Wall A",
        "host_name": "Basic Wall:MockUp Storage Wall"
      },
      {
        "predicate": "NEXT_TO",
        "object_type": "IfcDoor",
        "direction": "left",
        "confidence": 1.0
      },
      {
        "predicate": "NEXT_TO",
        "object_type": "IfcDoor",
        "direction": "right",
        "confidence": 1.0
      }
    ]
  },
  "raw_output": "{\"storey_name\": \"-1\", \"ifc_class\": \"IfcDoor\", \"space_name\": null, \"spatial_relations\": [{\"predicate\": \"FILLS\", \"object_type\": \"IfcWall\", \"confidence\": 1.0, \"object_material\": \"Interior Wall A\", \"host_name\": \"Basic Wall:MockUp Storage Wall\"}, {\"predicate\": \"NEXT_TO\", \"object_type\": \"IfcDoor\", \"direction\": \"left\", \"confidence\": 1.0}, {\"predicate\": \"NEXT_TO\", \"object_type\": \"IfcDoor\", \"direction\": \"right\", \"confidence\": 1.0}], \"position_context\": \"4th of 5 openings on the same wall\"}",
  "latency_ms": 16002.0,
  "status": "OK"
}
```

### 📊 Side-by-Side Comparison

| Feature / Attribute | Ground Truth | Raw Output | Status |
| :--- | :--- | :--- | :--- |
| **Storey Name** | `"-1"` | `"-1"` | ✅ Match |
| **IFC Class** | `"IfcDoor"` | `"IfcDoor"` | ✅ Match |
| **Space Name** | `null` | `null` | ✅ Match |
| **Relation 1 (FILLS)** | Object: `IfcWall`<br>Material: `Interior Wall A`<br>Host: `Basic Wall:MockUp Storage Wall` | Object: `IfcWall`<br>Material: `Interior Wall A`<br>Host: `Basic Wall:MockUp Storage Wall` | ✅ Match |
| **Relation 2 (NEXT\_TO)** | Object: `IfcDoor`<br>Direction: `left` | Object: `IfcDoor`<br>Direction: `left` | ✅ Match |
| **Relation 3 (NEXT\_TO)** | *(None)* | Object: `IfcDoor`<br>Direction: `right` | ❌ **Extra/Hallucinated** |
| **Position Context** | `"2nd of 2 openings on the same wall"` | `"4th of 5 openings on the same wall"` | ❌ **Mismatch** |

***Under previous prompt that need to be ammend***


## 2. LoRA_6 - new set of finetuning
see (mscd_demo/output/lora6_v2_ap_20260331/metrics/track_b2_summary.md)

## 3. Next
### - query planner redesign (in progress): previous: single 1-hop

| Group | GT-in-Pool | Top-10 | Top-1 | MRR@10 | Med Pool | Reduction | Gap vs Oracle 1-hop | Gap vs Oracle 2-hop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 Canonical | 100.0% | 25.0% | 1.7% | 0.0503 | 76 | 92.9% | 31 | 31 |
| G1 FullAug | 100.0% | 28.3% | 1.7% | 0.0564 | 76 | 92.9% | 31 | 31 |
| G2 FullAug LowLR | 100.0% | 28.3% | 1.7% | 0.0564 | 76 | 92.9% | 31 | 31 |
| G3 FullAug r32 | 100.0% | 28.3% | 1.7% | 0.0629 | 76 | 93.0% | 31 | 31 |
| Gemini AP | 56.7% | 11.7% | 0.0% | 0.0343 | 45 | 95.6% | 0 | 0 |

### - Oracle slicing test:
see (mscd_demo/output/lora6_v2_ap_20260331/oracle_ap_heldout/oracle_topology_summary.md)
```
    family = _topology_family(rels)
    if family == "singleton:CONNECTS_TO":
        return "U1"
    if family == "singleton:ADJACENT_TO":
        return "U2"
    if family == "paired:FILLS+NEXT_TO":
        return "U3"
    if family == "triad:FILLS+NEXT_TO+NEXT_TO":
        return "U4"
    if family == "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)":
        return "U5"
    return "U6"
```