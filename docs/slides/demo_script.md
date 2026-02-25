# Screenshot and gif to show demo UIUX:
Segment 1 — Show the System (False vs True)
MISS → HIT narrative: "cracked wall vs damaged door frame"

Case ID	Condition	Pool	Story
MISS	SYNTH_V3_028_AP_SK_028	LoRA / MC	1666 → 181	Site supervisor reports hairline cracks on a wall surface, sends photos — system cannot disambiguate among 181 identical walls on that floor
HIT	SYNTH_V3_072_AP_SK_072	LoRA / MC	1666 → 36	Walkthrough reveals door frame damage, photos taken — system correctly identifies the damaged door among 36
Chat context for MISS: "Found some hairline cracks on this surface. / See attached photo. / Check your inbox for the images." → Basic Wall:Generic - 200mm

Chat context for HIT: "Frame damage spotted during walkthrough. / Taking photos now. / Checking now." → Swedoor... door

Segment 2 — LoRA vs Prompt (same case, same condition)
Best pick: SYNTH_V3_001_AP_SK_001 / condition MC — most dramatic SSR contrast

System	Pool Result	Hit
LoRA	1666 → 1 candidate	✓
Prompt	1666 → 41 candidates	✗
Chat: "Crack on the slab surface, need assessment. / Copy that. / Waiting for access to the restricted area." → Floor:Concrete-Domestic 425mm

LoRA narrows the entire building (1666 elements) down to exactly 1 — the right floor slab. Prompt returns 41 slabs and picks the wrong one. Perfect visual contrast.

Alt pick if you want a different building type: SYNTH_V3_005_DXA_SK_005 / MA (DXA duplex)

LoRA: 258 → 2 (skylight), HIT ✓ — "Window inspection in progress"
Prompt: 258 → 100 (picks walls instead), MISS ✗
Segment 3 — Modality robustness: LoRA MB- HIT
Pick: SYNTH_V3_072_AP_SK_072 / LoRA / MB-

MB- = site photos available, 4D project context OFF
Pool: 1666 → 36 candidates, HIT
Chat: "Frame damage spotted during walkthrough. / Taking photos now." → door frame
Narrative: "Even with 4D task-status data turned off, the LoRA model uses photos + text alone to correctly identify the damaged door among 36 on this floor."

This is also the same case as Segment 1 HIT, so a good callback.

Segment 4 — Dense cluster robustness
Reality: all k≥40 cases = 0 hits across all 12 conditions. Frame this as two parts:

Part A — Show the hard limit (k=181): SYNTH_V3_028_AP_SK_028 / MC

Pool: 1666 → 181 walls — zero systems can disambiguate
Narrative: "This is an open challenge — when 181 identical walls exist on a floor and the description is vague, even LoRA cannot pinpoint the right one."
Part B — Show the boundary where LoRA succeeds (k=36): SYNTH_V3_084_AP_SK_084 / LoRA / MA

Pool: 1666 → 36 doors, HIT (note: Prompt misses this)
Chat: "Understood. / Roger. / Check your inbox for the images." → door
Narrative: "At k=36 — the highest density where any system succeeds — LoRA identifies the correct element. The LoRA advantage holds right up to the system's density limit."
Summary Table
Segment	Case ID	Profile	Cond	k	Pool Result	Hit
(1) Miss	SYNTH_V3_028_AP_SK_028	LoRA	MC	181	1666→181	✗
(1) Hit	SYNTH_V3_072_AP_SK_072	LoRA	MC	36	1666→36	✓
(2) LoRA	SYNTH_V3_001_AP_SK_001	LoRA	MC	1	1666→1	✓
(2) Prompt	SYNTH_V3_001_AP_SK_001	Prompt	MC	1	1666→41	✗
(3) MB-	SYNTH_V3_072_AP_SK_072	LoRA	MB-	36	1666→36	✓
(4) Hard limit	SYNTH_V3_028_AP_SK_028	LoRA	MC	181	1666→181	✗
(4) Best dense	SYNTH_V3_084_AP_SK_084	LoRA	MA	36	1666→36	✓
