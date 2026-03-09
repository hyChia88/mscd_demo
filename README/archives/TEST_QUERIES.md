# Test Queries for Agent Tools

This document contains example queries to test all 5 tools in the BIM inspection agent.

## Tool 1: `list_available_spaces()`
**Purpose:** Discover what rooms/floors/spaces are available in the IFC model

### Query to copy:
```
What spaces are available in this building?
```

**Expected behavior:**
- Agent calls `list_available_spaces()`
- Returns list of spaces like "Floor 0", "Floor 1" with element counts

---

## Tool 2: `get_elements_by_room(room_name)`
**Purpose:** Find all elements in a specific room/floor

### Query to copy:
```
Show me all elements on Floor 0
```

**Expected behavior:**
- Agent calls `get_elements_by_room("Floor 0")`
- Returns list of elements with names, types, and GUIDs

---

## Tool 3: `get_element_details(guid)`
**Purpose:** Get detailed properties of a specific element

### Query to copy:
```
Find all walls on Floor 0, then show me the details of the first wall
```

**Expected behavior:**
- Agent calls `get_elements_by_room("Floor 0")`
- Filters for wall elements
- Calls `get_element_details(guid)` for the first wall
- Returns detailed properties (material, fire rating, etc.)

---

## Tool 4: `generate_3d_view(guid)`
**Purpose:** Generate a visual render of an element

### Query to copy:
```
Show me a 3D view of the first door on Floor 0
```

**Expected behavior:**
- Agent calls `get_elements_by_room("Floor 0")`
- Filters for door elements
- Calls `generate_3d_view(guid)` for the first door
- Returns render path like `/server/renders/{guid}_inspection_view.png`

---

## Tool 5: `identify_element_visually(site_description_or_photo, candidate_guids_str)`
**Purpose:** Visual matching to identify which element matches a photo/description

### Query Option A - With Text Description:
```
I found a damaged cabinet on Floor 0. Can you help identify which one? Use visual matching to check cabinets against the description "white kitchen cabinet with wood countertop"
```

### Query Option B - With Photo File Path:
```
I have a photo of a damaged wall at data/site_photos/evidence_01.jpg. Can you find walls on Floor 0 and use visual matching to identify which wall this is?
```

**Expected behavior:**
- Agent calls `get_elements_by_room("Floor 0")`
- Filters for relevant elements (cabinets/walls)
- Extracts GUIDs from results
- Calls `identify_element_visually("description or file path", "guid1,guid2,guid3")`
- Returns best match with confidence score

---

## Complete Workflow Test
**Purpose:** Test all tools in one conversation

### Query to copy:
```
I need to inspect Floor 0. First, show me what's available in the building. Then list all elements on Floor 0. I'm particularly interested in walls - can you get details for one of them and generate a 3D view? Finally, I found a damaged cabinet - help me identify which one using visual matching with the description "brown wooden cabinet".
```

**Expected behavior:**
- Calls `list_available_spaces()`
- Calls `get_elements_by_room("Floor 0")`
- Filters for walls
- Calls `get_element_details(guid)` for a wall
- Calls `generate_3d_view(guid)` for the same wall
- Filters for cabinets
- Calls `identify_element_visually("brown wooden cabinet", "cabinet_guids")`

---

## Quick Individual Tests

Copy and paste these one at a time to test each tool:

1. **Test list_available_spaces:**
   ```
   What floors are in this building?
   ```

2. **Test get_elements_by_room:**
   ```
   List all elements on Floor 0
   ```

3. **Test get_element_details:**
   ```
   Get details for the first wall on Floor 0
   ```

4. **Test generate_3d_view:**
   ```
   Generate a 3D view of a door on Floor 0
   ```

5. **Test identify_element_visually (with description):**
   ```
   I need to identify a cabinet on Floor 0 that matches "white kitchen cabinet with wood countertop". Use visual matching.
   ```

6. **Test identify_element_visually (with photo):**
   ```
   Use visual matching with the photo at data/site_photos/evidence_01.jpg to identify a wall on Floor 0.
   ```

---

## Tips for Testing

- Start with simple queries to test individual tools
- The agent should automatically choose which tools to use
- Watch the tool calls in the output to see which tools are being invoked
- If a tool doesn't get called, try rephrasing the query to be more explicit
- For visual matching, make sure to mention "visual matching" or "identify which one"

---

## Common Issues

1. **Agent doesn't call list_available_spaces:**
   - Make the query more exploratory: "I don't know what's in this building, can you help?"

2. **Agent doesn't use visual matching:**
   - Be explicit: "Use visual matching to identify..." or "I have multiple candidates, which one matches..."

3. **Agent calls tools in wrong order:**
   - This is fine! The agent should figure out the workflow on its own
   - But if needed, you can guide it: "First list spaces, then get elements, then..."