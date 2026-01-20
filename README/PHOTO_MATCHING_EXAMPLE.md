# Photo-Based Visual Matching Example

## How to Use the Photo at data/site_photos/evidence_01.jpg

Based on the photo showing a damaged wall with cracking above white cabinets, here are test queries you can use:

---

## Example 1: Identify the Damaged Wall

### Query to copy:
```
I have a photo at data/site_photos/evidence_01.jpg showing a damaged wall with cracking. Find all walls on Floor 0 and use visual matching to identify which wall this is.
```

### What the agent should do:
1. Call `list_available_spaces()` or directly query Floor 0
2. Call `get_elements_by_room("Floor 0")`
3. Filter results for wall elements (IfcWall types)
4. Extract GUIDs of walls
5. Call `identify_element_visually("data/site_photos/evidence_01.jpg", "wall_guid1,wall_guid2,wall_guid3")`
6. Return the best matching wall with confidence score

---

## Example 2: Identify the Cabinets

### Query to copy:
```
The photo at data/site_photos/evidence_01.jpg shows white cabinets with a wood countertop. Use visual matching to identify which cabinet element this is on Floor 0.
```

### What the agent should do:
1. Call `get_elements_by_room("Floor 0")`
2. Filter for furniture/cabinet elements
3. Call `identify_element_visually("data/site_photos/evidence_01.jpg", "cabinet_guids")`
4. Return best match

---

## Example 3: Text Description Alternative (No Photo Path)

If you want to test with just a description instead of the file path:

### Query to copy:
```
I found a grey wall with cracking damage above white kitchen cabinets with wood countertop on Floor 0. Use visual matching to identify which wall this is.
```

This will use text-based CLIP matching instead of image-based matching.

---

## How It Works

The `identify_element_visually` tool:

1. **Accepts two types of input:**
   - **File path**: `data/site_photos/evidence_01.jpg` → Tool detects it's a file and processes accordingly
   - **Text description**: `"grey wall with cracking"` → Tool uses text directly for CLIP matching

2. **Processing:**
   - Gets candidate elements from IFC model
   - Extracts element names and types as descriptions
   - Uses CLIP embeddings to find semantic similarity
   - Returns best match with confidence score

3. **Current Implementation:**
   - Uses VisualAligner with CLIP model
   - Compares text embeddings (MVP version)
   - In production: Would also support image embeddings from actual photos

---

## File Path Formats

All these formats should work:

```
data/site_photos/evidence_01.jpg          ✓ Relative path
./data/site_photos/evidence_01.jpg        ✓ Explicit relative
/root/cmu/master_thesis/mscd_demo/data/site_photos/evidence_01.jpg  ✓ Absolute path
```

The tool uses `os.path.exists()` to check if the input is a file path.

---

## Testing Checklist

- [ ] Test with file path (Example 1)
- [ ] Test with text description (Example 3)
- [ ] Verify agent calls `get_elements_by_room` first
- [ ] Verify agent filters for correct element type
- [ ] Verify agent extracts GUIDs correctly
- [ ] Verify visual matching returns confidence score
- [ ] Check if CLIP model loads (or gracefully degrades to mock)