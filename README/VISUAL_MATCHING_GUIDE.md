# Visual Matching Implementation Guide

This guide shows you how to use the CLIP-based visual matching feature for matching site observations to BIM elements.

---

## Overview

**What it does**: Uses OpenAI's CLIP model to match text descriptions (or images) of site observations to BIM element descriptions using semantic similarity.

**Technology**:
- CLIP (Contrastive Language-Image Pre-training)
- Transforms text/images into 512-dimensional embeddings
- Computes cosine similarity between embeddings

**Use case**: Inspector describes damage → System finds matching BIM element

---

## Prerequisites

### 1. Check Requirements

The following should already be in `requirements.txt`:
```
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
```

### 2. Install Dependencies

```bash
# Make sure you're in the mscd_demo conda environment
conda activate mscd_demo

# Install if not already installed
pip install transformers torch numpy
```

**Note**: First run will download ~600MB CLIP model from HuggingFace.

---

## Method 1: Standalone Python Script

### Basic Usage

Create a test file `test_visual_matching.py`:

```python
from src.visual_matcher import VisualAligner

# Initialize the aligner (downloads model on first run)
print("Initializing Visual Aligner...")
aligner = VisualAligner()

# Scenario: Inspector found damage
site_observation = "Cracked grey concrete slab with exposed rebar"

# BIM elements in that area (from get_elements_by_room)
candidate_descriptions = [
    "Wooden kitchen cabinet with white finish",
    "Grey concrete structural slab, 200mm thick",
    "White painted drywall partition",
    "Ceramic tile flooring"
]

# Find best match
idx, score, match = aligner.find_best_match(
    site_observation,
    candidate_descriptions
)

# Display results
print(f"\n{'='*60}")
print(f"Site Observation: {site_observation}")
print(f"{'='*60}")
print(f"Best Match: {match}")
print(f"Match Index: {idx}")
print(f"Confidence Score: {score:.2%}")
print(f"{'='*60}")
```

### Run It

```bash
python test_visual_matching.py
```

**Expected Output**:
```
👁️ [VisualAligner] Initializing CLIP Model (Multimodal Embedding Space)...
✅ [VisualAligner] Model loaded on cpu
🔍 [VisualAligner] Computing Vector Similarity for: 'Cracked grey concrete slab with exposed rebar'

============================================================
Site Observation: Cracked grey concrete slab with exposed rebar
============================================================
Best Match: Grey concrete structural slab, 200mm thick
Match Index: 1
Confidence Score: 87.34%
============================================================
```

---

## Method 2: Integration with IFC Engine

### Complete Workflow: Site Photo → BIM Element

```python
from src.ifc_engine import IFCEngine
from src.visual_matcher import VisualAligner

# 1. Initialize systems
print("Loading IFC model...")
engine = IFCEngine("data/BasicHouse.ifc")

print("Loading visual matcher...")
aligner = VisualAligner()

# 2. Inspector reports damage location
location = "floor 0"
observation = "Damaged wooden cabinet near sink, brown finish"

print(f"\n📋 Site Report:")
print(f"  Location: {location}")
print(f"  Observation: {observation}")

# 3. Get all elements in that location
print(f"\n🔍 Querying BIM model for location '{location}'...")
elements = engine.find_elements_in_space(location)

# 4. Filter for furniture (since observation mentions "cabinet")
furniture = [el for el in elements if "cabinet" in el['name'].lower()]
print(f"  Found {len(furniture)} cabinet elements")

# 5. Create descriptions for visual matching
candidate_descriptions = []
candidate_guids = []

for item in furniture[:10]:  # Limit to 10 for performance
    # Create rich description from BIM data
    desc = f"{item['name']} ({item['type']})"
    candidate_descriptions.append(desc)
    candidate_guids.append(item['guid'])

# 6. Visual matching
print(f"\n🎯 Running visual similarity matching...")
idx, score, match = aligner.find_best_match(observation, candidate_descriptions)

# 7. Results
print(f"\n{'='*70}")
print(f"🎯 MATCH FOUND")
print(f"{'='*70}")
print(f"Element: {match}")
print(f"GUID: {candidate_guids[idx]}")
print(f"Confidence: {score:.2%}")
print(f"{'='*70}")

# 8. Get detailed properties
details = engine.get_element_properties(candidate_guids[idx])
print(f"\n📄 Element Details:")
print(details)
```

**Save as**: `examples/visual_matching_workflow.py`

**Run**:
```bash
python examples/visual_matching_workflow.py
```

---

## Method 3: Add as Agent Tool

### Enable Visual Matching in Agent

Currently `identify_element_visually` is defined but **NOT exported** in `agent_tools.py`.

**To enable it:**

1. Open `src/agent_tools.py`
2. Find line 110:
   ```python
   tools = [list_available_spaces, get_elements_by_room, get_element_details, generate_3d_view]
   ```

3. Add the visual matching tool:
   ```python
   tools = [list_available_spaces, get_elements_by_room, get_element_details, generate_3d_view, identify_element_visually]
   ```

4. Update system prompt in `prompts/system_prompt.yaml`:
   ```yaml
   Tools:
     1. list_available_spaces()
     2. get_elements_by_room(floor_name)
     3. get_element_details(guid)
     4. generate_3d_view(guid)
     5. identify_element_visually(site_photo_path, candidate_guids_str) - NEW!
   ```

### Using in Chat

```bash
python src/chat_cli.py
```

**Conversation:**
```
🧑 You: I found a damaged grey concrete slab. Can you help identify which one?

🤖 Agent: Let me search for concrete slabs on the available floors.
[Calls list_available_spaces()]
[Calls get_elements_by_room("floor 0")]
[Filters for IfcSlab type]

Found 3 concrete slabs. To identify the exact one, I can use visual matching.
Please provide a description or let me match against: "slab1_guid,slab2_guid,slab3_guid"

🧑 You: Use visual matching with description "grey cracked concrete" against those GUIDs

🤖 Agent: [Calls identify_element_visually("grey cracked concrete", "guid1,guid2,guid3")]
Best visual match: guid2 (confidence: 0.87)
```

---

## Advanced: Image-Based Matching (Future Enhancement)

Currently the system uses **text-to-text** matching. CLIP also supports **image-to-text** matching.

### Enhancement Plan

```python
class VisualAligner:
    # ... existing code ...

    def get_image_embedding(self, image_path: str):
        """
        Extract visual embedding from an uploaded site photo.
        """
        from PIL import Image

        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features / image_features.norm(dim=-1, keepdim=True)

    def match_photo_to_elements(self, photo_path: str, candidate_descriptions: list):
        """
        Match uploaded photo to BIM element descriptions.
        """
        print(f"📸 [VisualAligner] Analyzing photo: {photo_path}")

        photo_emb = self.get_image_embedding(photo_path)

        scores = []
        for candidate in candidate_descriptions:
            cand_emb = self.get_text_embedding(candidate)
            score = (photo_emb @ cand_emb.T).item()
            scores.append(score)

        best_idx = np.argmax(scores)
        return best_idx, scores[best_idx], candidate_descriptions[best_idx]
```

**Usage:**
```python
# Inspector uploads photo
photo_path = "uploads/damaged_cabinet.jpg"

# Match against BIM elements
idx, score, match = aligner.match_photo_to_elements(
    photo_path,
    candidate_descriptions
)
```

---

## Performance Optimization

### 1. Cache Embeddings

For repeated queries on the same BIM model:

```python
class VisualAlignerCached(VisualAligner):
    def __init__(self):
        super().__init__()
        self.embedding_cache = {}

    def get_text_embedding(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        emb = super().get_text_embedding(text)
        self.embedding_cache[text] = emb
        return emb
```

### 2. Batch Processing

Process multiple candidates at once:

```python
def get_text_embeddings_batch(self, texts: list):
    """Process multiple texts in one forward pass"""
    inputs = self.processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = self.model.get_text_features(**inputs)

    return text_features / text_features.norm(dim=-1, keepdim=True)
```

### 3. GPU Acceleration

If you have a GPU:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

The code already auto-detects GPU (line 12 in visual_matcher.py):
```python
self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Real-World Example Scenarios

### Scenario 1: Water Damage Identification

```python
from src.visual_matcher import VisualAligner

aligner = VisualAligner()

# Inspector's field note
observation = "Water stains on base cabinet, swollen wood near bottom"

# Get cabinets from BIM
candidates = [
    "M_Base Cabinet - Kitchen, oak finish",
    "M_Wall Cabinet - Kitchen, white painted",
    "M_Vanity Cabinet - Bathroom, laminate"
]

idx, score, match = aligner.find_best_match(observation, candidates)
print(f"Identified: {match} (confidence: {score:.0%})")
```

### Scenario 2: Crack Detection

```python
observation = "Vertical crack in grey concrete wall, about 2 meters high"

candidates = [
    "Yttervägg (Exterior Wall) - Concrete, 200mm thick",
    "Innerdörr (Interior Wall) - Drywall partition",
    "Floor slab - Concrete, 150mm thick"
]

idx, score, match = aligner.find_best_match(observation, candidates)
```

### Scenario 3: Multi-Element Comparison

```python
# Inspector found damage to multiple elements
observations = [
    "Cracked window frame, white color",
    "Damaged wooden door, brown finish",
    "Stained floor tile, ceramic"
]

# All candidates from a room
all_elements = engine.find_elements_in_space("floor 0")

for obs in observations:
    candidate_descs = [f"{el['name']} ({el['type']})" for el in all_elements]
    idx, score, match = aligner.find_best_match(obs, candidate_descs)

    print(f"Observation: {obs}")
    print(f"  → Match: {match} (confidence: {score:.0%})")
    print()
```

---

## Testing

### Unit Test

The file already includes a unit test (lines 52-62):

```bash
python src/visual_matcher.py
```

**Expected output**:
```
👁️ [VisualAligner] Initializing CLIP Model...
✅ [VisualAligner] Model loaded on cpu
🔍 [VisualAligner] Computing Vector Similarity for: 'Cracked grey concrete surface'
Input: Cracked grey concrete surface
Match: Grey concrete structural slab (Score: 0.8532)
```

### Integration Test

Create `tests/test_visual_integration.py`:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ifc_engine import IFCEngine
from src.visual_matcher import VisualAligner

def test_end_to_end():
    """Test complete workflow: IFC query → Visual matching"""

    # Setup
    engine = IFCEngine("data/BasicHouse.ifc")
    aligner = VisualAligner()

    # Query BIM
    elements = engine.find_elements_in_space("floor 0")
    assert len(elements) > 0, "Should find elements on floor 0"

    # Visual matching
    observation = "Grey concrete wall"
    candidates = [f"{el['name']} ({el['type']})" for el in elements[:5]]

    idx, score, match = aligner.find_best_match(observation, candidates)

    # Assertions
    assert 0 <= idx < len(candidates), "Index should be valid"
    assert 0 <= score <= 1, "Score should be between 0 and 1"
    assert match in candidates, "Match should be from candidates"

    print("✅ Integration test passed!")
    print(f"   Best match: {match} ({score:.0%})")

if __name__ == "__main__":
    test_end_to_end()
```

Run:
```bash
python tests/test_visual_integration.py
```

---

## Troubleshooting

### Issue: Model download fails

**Solution**:
```bash
# Manual download
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
```

### Issue: Out of memory

**Solution**: Use smaller model or reduce batch size
```python
# In visual_matcher.py, line 9:
self.model_id = "openai/clip-vit-base-patch16"  # Even smaller
```

### Issue: Low accuracy

**Improve descriptions**:
```python
# Bad: Generic description
"Cabinet"

# Good: Rich description
"M_Base Cabinet - Kitchen, oak wood finish, brown color, 60cm width"
```

### Issue: Slow performance

**Solutions**:
1. Use GPU if available
2. Cache embeddings
3. Batch process candidates
4. Reduce candidate list size

---

## Next Steps

1. **Test basic matching**:
   ```bash
   python src/visual_matcher.py
   ```

2. **Try the workflow example**:
   Create and run `examples/visual_matching_workflow.py`

3. **Enable in agent** (optional):
   Add `identify_element_visually` to tools export

4. **Enhance with images**:
   Implement `get_image_embedding` for photo uploads

5. **Optimize**:
   Add caching and batch processing

---

## Summary

| Feature | Status | Command |
|---------|--------|---------|
| Text-to-text matching | ✅ Working | `python src/visual_matcher.py` |
| IFC integration | ✅ Working | Create workflow script |
| Agent integration | ⚠️ Defined but not exported | Edit `agent_tools.py` line 110 |
| Image-to-text matching | ❌ Not implemented | Enhance `VisualAligner` class |
| Caching | ❌ Not implemented | Add `VisualAlignerCached` |

The core functionality is **ready to use** - just run the examples above!
