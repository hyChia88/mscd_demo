# Evaluation Plots - Quick Guide

## Automatic Timestamped Directories

The plot generator now automatically creates timestamped directories to prevent overwriting previous results.

### Format

```
logs/plots/<timestamp>_<profile_name>/
```

**Example Structure**:
```
logs/plots/
├── 20260210_182748_v2_prompt/
│   ├── 1_accuracy_by_condition.png
│   ├── 2_search_space_reduction.png
│   ├── 3_constraints_parse_rate.png
│   ├── 4_image_parse_timing.png        (if VLM data available)
│   ├── 5_vision_impact.png             (if comparison data available)
│   └── 6_per_case_heatmap.png
└── 20260210_183055_v2_prompt/
    ├── 1_accuracy_by_condition.png
    ├── 2_search_space_reduction.png
    └── ...
```

---

## Usage

### Auto-timestamped Output (Default)

```bash
conda run -n mscd_demo python script/generate_plots.py --latest
# Output: logs/plots/20260210_182748_v2_prompt/
```

### Custom Output Directory

```bash
conda run -n mscd_demo python script/generate_plots.py \
  --latest \
  --output my_custom_plots
# Output: my_custom_plots/
```

### Compare Before/After

```bash
conda run -n mscd_demo python script/generate_plots.py \
  --traces logs/evaluations/new_pipeline/traces_*.jsonl \
  --before logs/evaluations/old_pipeline/traces_*.jsonl
# Output: logs/plots_<timestamp>_<profile>/
```

---

## Benefits

✅ **No Overwriting**: Each run gets its own directory
✅ **Traceable**: Timestamp shows when plots were generated
✅ **Profile Tagged**: Directory name includes the profile used (v2_prompt, v1_baseline, etc.)
✅ **Easy Comparison**: Keep multiple plot sets side-by-side

---

## Example Workflow

```bash
# Run evaluation
conda run -n mscd_demo python script/run.py \
  --profile v2_prompt \
  --cases test_subset.jsonl \
  --condition B1

# Generate plots (auto-timestamped)
conda run -n mscd_demo python script/generate_plots.py --latest
# Output: logs/plots/20260210_183000_v2_prompt/

# Run again with different condition
conda run -n mscd_demo python script/run.py \
  --profile v2_prompt \
  --cases test_subset.jsonl \
  --condition C2

# Generate plots (new timestamp)
conda run -n mscd_demo python script/generate_plots.py --latest
# Output: logs/plots/20260210_183500_v2_prompt/

# Now you have both sets preserved!
ls logs/plots/*/
# logs/plots/20260210_183000_v2_prompt/
# logs/plots/20260210_183500_v2_prompt/
```

---

## Finding Your Plots

```bash
# List all plot directories
ls -ld logs/plots/*/

# View latest plots
ls -lh $(ls -td logs/plots/*/ | head -1)

# Open latest plot directory
cd $(ls -td logs/plots/*/ | head -1) && ls -lh
```

---

## For Thesis

When selecting plots for your thesis, the timestamp helps you track which evaluation run produced which results.

**Recommended**: After generating final thesis plots, copy them to a dedicated folder:

```bash
# After final evaluation run
conda run -n mscd_demo python script/generate_plots.py --latest

# Copy to thesis folder
mkdir -p thesis_figures
cp logs/plots/<your_timestamp>_<profile>/*.png thesis_figures/

# Or use the latest automatically
cp $(ls -td logs/plots/*/ | head -1)/*.png thesis_figures/
```

Then reference from LaTeX:
```latex
\includegraphics[width=0.8\textwidth]{thesis_figures/2_search_space_reduction.png}
```
