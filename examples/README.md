# Examples Directory

Practical examples demonstrating how to use the BIM Inspection Agent features.

## Available Examples

### 1. `test_visual_matching.py` - Visual Matching Basics

**What it does**: Demonstrates CLIP-based visual matching without IFC integration

**Includes**:
- Demo 1: Basic text-to-text matching
- Demo 2: Multiple site observations
- Demo 3: Detailed similarity scoring
- Demo 4: Semantic understanding test

**Run**:
```bash
python examples/test_visual_matching.py
```

**Expected output**:
```
DEMO 1: Basic Visual Matching
======================================================================
📋 Site Observation:
   'Cracked grey concrete slab with exposed rebar'

🔍 Searching among 4 BIM elements:
   1. Wooden kitchen cabinet with white finish
   2. Grey concrete structural slab, 200mm thick
   3. White painted drywall partition
   4. Ceramic tile flooring

✅ RESULT:
   Best Match: Grey concrete structural slab, 200mm thick
   Match Index: 2
   Confidence: 87.34%
```

**Use when**:
- Testing visual matching without IFC file
- Understanding how CLIP embeddings work
- Comparing similarity scores
- Testing semantic understanding

---

### 2. `visual_matching_workflow.py` - Complete IFC Integration

**What it does**: Full workflow from site damage report to BIM element identification

**Workflow steps**:
1. Initialize IFC engine and visual matcher
2. Receive site damage report
3. Query BIM model by location
4. Filter relevant element types
5. Run visual matching
6. Get detailed properties
7. Generate recommendations

**Run**:
```bash
python examples/visual_matching_workflow.py
```

**Expected output**:
```
COMPLETE VISUAL MATCHING WORKFLOW
Site Damage Report → BIM Element Identification
======================================================================

📦 STEP 1: Initializing Systems
──────────────────────────────────────────────────────────────────────
  Loading IFC model...
  Loading visual matcher...
✅ Systems ready

📋 STEP 2: Site Damage Report
──────────────────────────────────────────────────────────────────────
  Location: floor 0
  Observation: Water damage on brown wooden base cabinet near sink area

🔍 STEP 3: Querying BIM Model
──────────────────────────────────────────────────────────────────────
  Searching for elements in 'floor 0'...
  ✅ Found 123 total elements
  Filtering for cabinets...
  ✅ Found 8 cabinet/furniture elements

🎯 STEP 4: Preparing Visual Matching
──────────────────────────────────────────────────────────────────────
  Candidate elements:
    1. M_Base Cabinet (IfcFurniture)
    2. M_Wall Cabinet (IfcFurniture)
    ...

🔬 STEP 5: Running Visual Similarity Analysis
──────────────────────────────────────────────────────────────────────
  Comparing observation to 8 candidates...
  ✅ Match found!

📊 STEP 6: Match Results
======================================================================

🎯 IDENTIFIED ELEMENT:
  Name:        M_Base Cabinet
  Type:        IfcFurniture
  GUID:        abc123xyz...
  Confidence:  89.23%

📄 STEP 7: Fetching Detailed Properties
──────────────────────────────────────────────────────────────────────
{
  'GlobalId': 'abc123xyz...',
  'Name': 'M_Base Cabinet',
  'Type': 'IfcFurniture',
  'PredefinedType': 'Kitchen Cabinet'
}

📝 STEP 8: Summary & Recommended Actions
======================================================================

✅ Successfully identified damaged element:
   Element: M_Base Cabinet
   GUID: abc123xyz...
   Match confidence: 89.23%

📋 Recommended next actions:
   1. Create work order for GUID: abc123xyz...
   2. Schedule inspection of M_Base Cabinet
   3. Document damage with photos
   4. Generate 3D view for verification
```

**Use when**:
- Testing complete end-to-end workflow
- Demonstrating to stakeholders
- Understanding integration points
- Creating documentation

---

## Quick Start

### Prerequisites

```bash
# Activate environment
conda activate mscd_demo

# Ensure dependencies are installed
pip install transformers torch numpy
```

### Test Basic Matching (No IFC Required)

```bash
python examples/test_visual_matching.py
```

This will download the CLIP model (~600MB) on first run.

### Test Complete Workflow (Requires IFC File)

```bash
# Make sure BasicHouse.ifc exists
ls data/BasicHouse.ifc

# Run workflow
python examples/visual_matching_workflow.py
```

---

## Customizing Examples

### Modify Site Observations

Edit the observation text in either script:

```python
# In test_visual_matching.py, line 28
site_observation = "Your custom observation here"

# In visual_matching_workflow.py, line 60
observation = "Your custom observation here"
```

### Change Element Candidates

```python
# In test_visual_matching.py, line 31
candidate_descriptions = [
    "Your element 1",
    "Your element 2",
    "Your element 3"
]
```

### Query Different Location

```python
# In visual_matching_workflow.py, line 59
location = "floor 1"  # or any space from your IFC
```

---

## Performance Notes

### First Run
- Downloads CLIP model (~600MB)
- Takes 1-2 minutes to initialize
- Model cached for future runs

### Subsequent Runs
- Loads model from cache
- Initialization takes ~10 seconds
- Each matching operation takes ~1 second

### GPU Acceleration
If you have a CUDA-compatible GPU:
```python
# Check if GPU is available
import torch
print(torch.cuda.is_available())
```

The code automatically uses GPU if available (see `visual_matcher.py` line 12).

---

## Troubleshooting

### Model Download Issues

```bash
# Manual download
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
```

### IFC File Not Found

```bash
# Check if file exists
ls -la data/BasicHouse.ifc

# Run inspection to see what's in your IFC
python src/inspect_ifc.py
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade transformers torch numpy ifcopenshell
```

### Low Matching Accuracy

Improve descriptions:
- Add more details (color, material, size)
- Use specific terminology
- Include visual characteristics

Example:
```python
# Bad
"Cabinet"

# Good
"M_Base Cabinet - Kitchen, oak wood, brown finish, 60cm width"
```

---

## Next Steps

1. **Run basic demo**:
   ```bash
   python examples/test_visual_matching.py
   ```

2. **Run IFC workflow**:
   ```bash
   python examples/visual_matching_workflow.py
   ```

3. **Create custom example**:
   - Copy one of the example files
   - Modify observations and candidates
   - Test with your own IFC file

4. **Integrate with agent**:
   - See `VISUAL_MATCHING_GUIDE.md` for agent integration
   - Enable `identify_element_visually` tool

---

## Additional Resources

- [VISUAL_MATCHING_GUIDE.md](../VISUAL_MATCHING_GUIDE.md) - Comprehensive guide
- [WORKFLOW_EXAMPLE.md](../WORKFLOW_EXAMPLE.md) - General workflow examples
- [src/visual_matcher.py](../src/visual_matcher.py) - Source code
- [src/agent_tools.py](../src/agent_tools.py) - Tool integration

---

## Example Use Cases

### 1. Water Damage Investigation
```python
observation = "Water stains on base cabinet, swollen wood near bottom"
# → Matches to specific cabinet GUID for work order
```

### 2. Structural Inspection
```python
observation = "Vertical crack in grey concrete wall, approximately 2 meters"
# → Identifies specific wall element for structural engineer
```

### 3. Maintenance Planning
```python
observations = [
    "Cracked window frame, white color",
    "Damaged wooden door, brown finish",
    "Stained floor tile, ceramic"
]
# → Batch process multiple damage reports
```

### 4. Quality Control
```python
observation = "Defective cabinet door, doesn't close properly"
# → Match to BIM element → Extract specifications → Verify compliance
```

---

## Contributing

To add new examples:

1. Create new file in `examples/` directory
2. Follow naming convention: `verb_noun.py`
3. Include docstring with usage instructions
4. Add error handling
5. Update this README

Example template:
```python
#!/usr/bin/env python3
"""
Brief description

Usage:
    python examples/your_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visual_matcher import VisualAligner
# ... your code ...

if __name__ == "__main__":
    # Your example code
    pass
```
