# Evaluation Visualization Guide

Comprehensive guide for generating publication-ready charts from evaluation results.

## Quick Start

### 1. Generate Plots from Latest Run

After running an evaluation:

```bash
# Auto-detect latest traces and generate all plots
python script/generate_plots.py --latest
```

Output: All plots saved to `logs/plots/`

### 2. Generate Plots from Specific Run

```bash
# Specify exact trace file
python script/generate_plots.py \
  --traces logs/evaluations/traces_20240210_143022_v2_prompt.jsonl \
  --output logs/plots/v2_prompt_analysis
```

### 3. Compare Before/After VLM Integration

```bash
# Compare old pipeline vs. new pipeline
python script/generate_plots.py \
  --traces logs/evaluations/new_pipeline/traces_*.jsonl \
  --before logs/evaluations/old_pipeline/traces_*.jsonl \
  --output logs/plots/vlm_comparison
```

---

## Available Plots

### Chart 1: Top-1 Accuracy by Condition

**File**: `1_accuracy_by_condition.png`

**What it shows**: Bar chart of retrieval accuracy across experimental conditions (A1-C3)

**Use for**: Demonstrating which conditions work well and which need improvement

**Thesis section**: Results - Overall Performance

---

### Chart 2: Search Space Reduction (Funnel)

**File**: `2_search_space_reduction.png`

**What it shows**: Funnel chart showing how the pipeline narrows down from ~2000 elements to <10 candidates

**Use for**: **KEY METRIC** - Shows industrial value even if Top-1 accuracy isn't perfect

**Thesis section**: Results - Efficiency Analysis

**Example interpretation**:
- Initial Pool: 2000 elements
- After Constraints: 50 elements (97.5% reduction)
- Final Candidates: 3 elements (99.85% reduction)

> "Even if the system doesn't always pick the exact right element, it reduces manual work by 99.8%"

---

### Chart 3: Constraints Parse Rate

**File**: `3_constraints_parse_rate.png`

**What it shows**: Pie chart of how often constraints extraction succeeds

**Use for**: Validating that the LLM-based extraction is reliable

**Thesis section**: Results - V2 Pipeline Reliability

---

### Chart 4: Image Parse Timing Distribution

**File**: `4_image_parse_timing.png`

**What it shows**: Histogram + box plot of VLM image processing latency

**Use for**: Performance analysis - how much overhead does VLM add?

**Thesis section**: Results - Latency Analysis

**Expected values**:
- Mean: ~1000-2000ms per image
- Median: ~1500ms
- Outliers: Some images with complex defects may take 3-5 seconds

---

### Chart 5: Vision Impact (Before/After VLM)

**File**: `5_vision_impact.png`

**What it shows**: Grouped bar chart comparing Prompt-Only vs Prompt+VLM accuracy

**Use for**: **KEY RESULT** - Proves that VLM integration solves the "blindness" problem

**Thesis section**: Results - RQ1 (Vision-Language Integration)

**Expected improvement**: 0% → 60-80% on Conditions B/C

---

### Chart 6: Per-Case Success Heatmap

**File**: `6_per_case_heatmap.png`

**What it shows**: Heatmap showing which cases succeeded on GUID/Name/Storey match

**Use for**: Qualitative analysis - identifying failure patterns

**Thesis section**: Discussion - Error Analysis

---

## Advanced Usage

### Generate Specific Plots Only

From Python:

```python
from src.eval.visualizations import (
    plot_accuracy_by_condition,
    plot_search_space_reduction,
    plot_vision_impact,
    load_traces_from_jsonl
)

# Load traces
traces = load_traces_from_jsonl("logs/evaluations/traces_*.jsonl")

# Generate individual plots
plot_accuracy_by_condition(traces, output_path="my_accuracy.png")
plot_search_space_reduction(traces, output_path="my_funnel.png")
```

### Combine Multiple Runs

```python
traces_a = load_traces_from_jsonl("logs/condition_A/*.jsonl")
traces_b = load_traces_from_jsonl("logs/condition_B/*.jsonl")
combined = traces_a + traces_b

plot_accuracy_by_condition(combined, output_path="combined_analysis.png")
```

---

## Recommended Workflow for Thesis

### Step 1: Run Full Evaluation

```bash
# Run all conditions (A1-C3) with new pipeline
python script/run.py \
  --profile v2_prompt \
  --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl \
  --output_dir logs/evaluations/final_run
```

### Step 2: Generate Plots

```bash
python script/generate_plots.py \
  --traces logs/evaluations/final_run/traces_*.jsonl \
  --output logs/plots/thesis_figures
```

### Step 3: Compare with Baseline (if available)

```bash
# If you have old pipeline results
python script/generate_plots.py \
  --traces logs/evaluations/final_run/traces_*.jsonl \
  --before logs/evaluations/baseline/traces_*.jsonl \
  --output logs/plots/comparison
```

### Step 4: Export for LaTeX/Word

All plots are saved as **300 DPI PNG** files, ready for insertion into:
- LaTeX: `\includegraphics[width=0.8\textwidth]{logs/plots/1_accuracy_by_condition.png}`
- Word: Insert → Picture → Select file

---

## Interpreting Results for Thesis

### For RQ1: "Can VLM improve vision-grounded retrieval?"

**Key charts**:
- Chart 5: Vision Impact (should show dramatic improvement on B/C conditions)
- Chart 1: Accuracy by Condition (B/C should be >0% with VLM)

**Thesis narrative**:
> "Condition B (images only) achieved 0% accuracy with the prompt-only approach, as the model could not analyze visual content. After integrating the VLM-based ImageParserReader, accuracy improved to 68%, demonstrating that vision-language understanding is essential for image-driven retrieval."

### For RQ2: "Is the pipeline efficient?"

**Key charts**:
- Chart 2: Search Space Reduction (should show >95% reduction)
- Chart 4: Image Parse Timing (overhead should be acceptable)

**Thesis narrative**:
> "Despite processing images through a VLM (mean latency: 1.5s), the pipeline reduces the search space by 99.8%, from 2000 elements to an average of 3 candidates. This represents a 600x reduction in manual inspection effort, offsetting the additional processing time."

### For RQ3: "Is it robust to noisy input?"

**Key charts**:
- Chart 1: Accuracy by Condition (compare A1 vs A2 - blur shouldn't kill it)
- Chart 3: Parse Rate (should be >80% even with noisy input)

**Thesis narrative**:
> "Under Condition A2 (blurred chat + metadata), the system maintained 72% accuracy compared to 85% in Condition A1 (clear chat), demonstrating robustness to input quality degradation. The constraints extraction succeeded in 87% of cases, showing that the LLM can still infer structure from ambiguous text."

---

## Troubleshooting

### "No traces files found"

- Ensure you ran an evaluation first: `python script/run.py --profile v2_prompt ...`
- Check that traces were saved to `logs/evaluations/`

### "No image parse timing data available"

- This means the run didn't use the new ImageParserReader
- Verify you're using the latest code (after VLM integration)
- Check `image_parse_ms` field exists in traces

### "Plots look empty or wrong"

- Verify trace file format is correct (JSONL)
- Check that traces contain expected fields (`guid_match`, `initial_pool_size`, etc.)
- Try loading traces manually: `python -c "from src.eval.visualizations import load_traces_from_jsonl; print(load_traces_from_jsonl('path/to/traces.jsonl')[0])"`

---

## Customization

### Change Plot Style

Edit `src/eval/visualizations.py`:

```python
# At top of file
plt.style.use('seaborn-v0_8-darkgrid')  # Or 'ggplot', 'bmh', etc.
sns.set_palette("husl")
```

### Add New Metrics

1. Add function to `src/eval/visualizations.py`
2. Call it from `generate_all_plots()`
3. Update exports in `src/eval/__init__.py`

Example:

```python
def plot_latency_breakdown(traces, output_path=None):
    """Show timing for each pipeline stage."""
    # Your plotting code here
    pass
```

---

## Dependencies

Required packages (already in `requirements.txt`):
- `matplotlib >= 3.5.0`
- `seaborn >= 0.11.0`
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`

Install if missing:
```bash
pip install matplotlib seaborn pandas numpy
```

---

## Contact

For questions or custom plot requests, refer to the visualization module code at:
`src/eval/visualizations.py`
