# Example Workflow Guide

This guide demonstrates how to use the AI-powered BIM inspection system step-by-step.

---

## Prerequisites

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up API Key**
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

3. **Prepare IFC File**
Place your IFC building model at:
```
data/BasicHouse.ifc
```

---

## Workflow 1: Inspect Your IFC Model

Before running the AI agent, understand your building model structure:

```bash
python src/inspect_ifc.py
```

**Expected Output:**
```
🏗️  Loading IFC Model: BasicHouse.ifc...
⚙️  Building Semantic Graph Index...
✅  Graph Index Ready: 2 groups indexed.

📊 Statistics: Found 2 spaces (Rooms/Spaces)

🏠 Room: 'floor 0'
   └── Contains 123 elements
       - [IfcWall] Yttervägg (GUID: 1a2b3c...)
       - [IfcDoor] Innerdörr (GUID: 4d5e6f...)
       - [IfcFurniture] M_Base Cabinet (GUID: 7g8h9i...)
       ...
```

**What this tells you:**
- Your model has 2 floors: "Floor 0" and "Floor 1"
- Floor 0 contains walls, doors, windows, furniture
- You now know what room names to use in your queries

---

## Workflow 2: Run a Simple Test Scenario

### Step 1: Create a Test Scenario File

Edit `test.yaml`:

```yaml
- name: "Kitchen Cabinet Inspection"
  description: "Inspector reports damage to kitchen cabinets on Floor 0"
  input: |
    Site Report:
    Location: Floor 0, Kitchen Area
    Observation: Found 3 base cabinets with water damage near sink.

    Question: Can you identify all kitchen cabinets on Floor 0 and provide their details?
```

### Step 2: Run the Agent

```bash
python src/main.py
```

### Step 3: Observe Agent Workflow

The agent will:

1. **Parse the Report**
   ```
   📋 Scenario 1: Kitchen Cabinet Inspection
   📥 Input:
   Site Report:
   Location: Floor 0, Kitchen Area
   ...
   ```

2. **Extract Location** → Identifies "Floor 0"

3. **Query the IFC Model** → Calls `get_elements_by_room("floor 0")`

4. **Filter Results** → Searches for "cabinet" keywords

5. **Get Details** → Calls `get_element_details(guid)` for each cabinet

6. **Generate Report**
   ```
   📤 Final Response:
   Found 8 kitchen cabinets on Floor 0:

   1. M_Base Cabinet (GUID: abc123)
      - Type: IfcFurniture
      - ObjectType: Kitchen Cabinet

   2. M_Wall Cabinet (GUID: def456)
      - Type: IfcFurniture
      - ObjectType: Kitchen Cabinet
   ...
   ```

---

## Workflow 3: Multi-Scenario Testing

### Test Multiple Inspection Cases

Edit `test.yaml` with multiple scenarios:

```yaml
# Scenario 1: Find Elements by Room
- name: "Exterior Wall Inspection"
  description: "Check all exterior walls for fire rating compliance"
  input: |
    Inspector needs to verify all exterior walls (Yttervägg) on Floor 0
    have proper fire rating. Please list all exterior walls and their properties.

# Scenario 2: Specific Element Type
- name: "Door Hardware Check"
  description: "Inventory all doors for hardware replacement"
  input: |
    Facilities team needs inventory of all doors on Floor 0.
    Please provide door count, names, and GUIDs.

# Scenario 3: Furniture Audit
- name: "Furniture Inventory"
  description: "Count furniture for office planning"
  input: |
    We need a complete furniture inventory for Floor 0 including:
    - Beds
    - Tables
    - Cabinets
    - Chairs
```

Run all scenarios:
```bash
python src/main.py
```

The agent will process each scenario sequentially with a 3-second pause between them.

---

## Workflow 4: Interactive Testing (Python REPL)

For ad-hoc queries, use Python interactively:

```python
from src.ifc_engine import IFCEngine

# Load the model
engine = IFCEngine("data/BasicHouse.ifc")

# Query 1: Find all elements in a room
kitchen_items = engine.find_elements_in_space("floor 0")
print(f"Found {len(kitchen_items)} items on Floor 0")

# Query 2: Filter for specific type
cabinets = [item for item in kitchen_items if "cabinet" in item['name'].lower()]
print(f"Found {len(cabinets)} cabinets")

# Query 3: Get detailed properties
for cabinet in cabinets[:3]:
    details = engine.get_element_properties(cabinet['guid'])
    print(details)
```

---

## Workflow 5: Visual Matching (Advanced)

**Note:** Requires CLIP model (large download ~600MB)

### Use Case: Match Site Photo to BIM Element

```python
from src.visual_matcher import VisualAligner

# Initialize (downloads model on first run)
aligner = VisualAligner()

# Inspector's observation
site_observation = "Cracked grey concrete slab with exposed rebar"

# BIM elements found in that area
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

print(f"Site Observation: {site_observation}")
print(f"Best Match: {match}")
print(f"Confidence Score: {score:.2%}")
```

**Expected Output:**
```
🔍 [VisualAligner] Computing Vector Similarity...
Site Observation: Cracked grey concrete slab with exposed rebar
Best Match: Grey concrete structural slab, 200mm thick
Confidence Score: 87.3%
```

---

## Workflow 6: Generate 3D Renders (Optional)

**Note:** Requires Blender installation

### Setup Blender Path

Edit `src/blender_service.py`:
```python
# For Windows
BLENDER_EXE = "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"

# For Mac
BLENDER_EXE = "/Applications/Blender.app/Contents/MacOS/Blender"

# For Linux
BLENDER_EXE = "/usr/bin/blender"
```

### Render a Specific Element

```python
from src.blender_service import run_blender_render

ifc_path = "data/BasicHouse.ifc"
target_guid = "1a2b3c4d5e6f"  # GUID from your inspection

output_path = run_blender_render(ifc_path, target_guid)
print(f"Render saved to: {output_path}")
```

---

## Common Use Cases

### 1. **Compliance Audits**
```yaml
input: |
  Check all IfcWall elements on Floor 0 for fire rating compliance.
  Report any walls missing PredefinedType or ObjectType properties.
```

### 2. **Defect Localization**
```yaml
input: |
  Inspector found water damage in the northwest corner of Floor 0.
  Identify all furniture and wall elements in that area.
```

### 3. **Quantity Takeoff**
```yaml
input: |
  Count all doors on Floor 0 and group by type (interior vs exterior).
```

### 4. **Asset Tracking**
```yaml
input: |
  List all furniture items (M_* prefix) for asset management.
  Include Name, Type, and GUID for each item.
```

---

## Understanding Agent Behavior

### What the Agent Does:

1. **Extracts Keywords** from unstructured text
   - "Floor 0" → Calls `get_elements_by_room("floor 0")`
   - "cabinets" → Filters results for matching names/types

2. **Semantic Filtering** within results
   - Searches for element names containing keywords
   - Matches element types (IfcWall, IfcDoor, etc.)

3. **Detail Retrieval** for compliance checks
   - Calls `get_element_details(guid)` for each match
   - Extracts Material, Fire Rating, ObjectType

4. **Structured Output** in readable format

### What the Agent Does NOT Do:

- ❌ Execute geometric calculations (area, volume)
- ❌ Modify the IFC model
- ❌ Generate new GUIDs or elements
- ❌ Make compliance decisions (only reports data)

---

## Troubleshooting

### Issue: "IFC file not found"
**Solution:** Verify file path in `src/agent_tools.py:8`
```python
IFC_PATH = os.path.join(BASE_DIR, "data", "BasicHouse.ifc")
```

### Issue: "No elements found"
**Solution:** Run `inspect_ifc.py` to see actual room names in your model

### Issue: Agent returns empty results
**Solution:** Check your query uses exact floor names from spatial index:
```python
# Wrong: "first floor", "ground floor"
# Correct: "floor 0", "floor 1"
```

### Issue: CLIP model fails to load
**Solution:** Visual matching is optional. The system works without it.
```python
# Disable in agent_tools.py
tools = [get_elements_by_room, get_element_details, generate_3d_view]
# Don't include: identify_element_visually
```

---

## Next Steps

1. **Customize for Your IFC Model**
   - Update file path in `agent_tools.py`
   - Run `inspect_ifc.py` to understand structure
   - Adjust system prompt in `main.py` with your floor names

2. **Add Custom Tools**
   - Extend `agent_tools.py` with new functions
   - Example: `get_elements_by_type()`, `calculate_area()`

3. **Integrate with Web UI**
   - Wrap `main.py` logic in FastAPI endpoint
   - Accept user queries via HTTP POST
   - Return JSON responses

4. **Production Deployment**
   - Add error handling and logging
   - Implement caching for IFC parsing
   - Use async/await for better performance
