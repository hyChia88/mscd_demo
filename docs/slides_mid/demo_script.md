# Screenshot and gif to show demo UIUX:
(1) Show system & LoRA vs Prompt: **Diff btwn LoRA & Prompt**
**case: SYNTH_V3_049_DXA_SK_049** LoRA MC- & Prompt MC 
    1. Show T in LoRA MC-
        LoRA	1666 → 1 candidate	✓
    2. Show F in Prompt MC
        Prompt	1666 → 41 candidates	✗

(2) Modality robustness:  HIT **Adding Photos help**
**case: SYNTH_V3_072_AP_SK_072** LoRA MB- 
    1. Show ✓ LoRA MB- 
    MB- = site photos available, 4D project context OFF
    Pool: 1666 → 36 candidates, HIT
    Chat: "Frame damage spotted during walkthrough. / Taking photos now." → door frame
    Narrative: "Even with 4D task-status data turned off, the LoRA model uses photos + text alone to correctly identify the damaged door among 36 on this floor."
**SYNTH_V3_019_BH_SK_019** LoRA/MB- 
    1. Show LoRA/MB- 	BH window, photos + no 4D → still hits

(3) Dense cluster robustness **Works in Dense cluster**
**case: SYNTH_V3_084_AP_SK_084** ✓ LoRA MA 
    1. Show ✓ LoRA MA 1666→36
    2. Show ✗ Prompt