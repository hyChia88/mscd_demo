Fix 1 (Highest ROI, ~1 hour): position_context for CONNECTS_TO via connection_degree
Every CONNECTS_TO skeleton already has connection_degree (range 2–22, mean 6.4). This maps directly to a descriptive position_context that breaks the 137× repeated fingerprint:


connection_degree=8  →  position_context: "wall junction node with 8 connections"
connection_degree=2  →  position_context: "wall end segment with 2 connections"
connection_degree=13 →  position_context: "high-degree hub wall with 13 connections"
Impact: Splits the worst fingerprint cluster (n=137, currently identical) into ~10 distinct degree groups. No new mining, no re-rendering — just label enrichment on existing JSONL.

Fix 2 (Medium ROI, ~1 hour): position_context for ADJACENT_TO via distance_mm
Every ADJACENT_TO skeleton already has distance_mm (e.g., 206mm). This gives a proximity descriptor:


distance_mm=206  →  position_context: "nearest adjacent door at 206mm"
distance_mm=890  →  position_context: "adjacent door at 890mm (mid-range)"
Fix 3 (Medium ROI, ~2 hours): Soft confidence labels
Change ambiguous cases to conf=0.7–0.8 — cases where the floorplan crop is partial, CONNECTS_TO walls have high connection_degree (many near-equal candidates), or ADJACENT_TO distance > 1000mm.