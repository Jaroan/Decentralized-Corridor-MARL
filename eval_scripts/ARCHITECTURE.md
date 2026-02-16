# Evaluation Architecture - Clean Separation

This document explains how the evaluation system cleanly separates debugging/training from formal evaluations.

## Problem Solved

❌ **Old approach**: CSV files in `graph_mpe_runner.py` get polluted with debugging data
✅ **New approach**: Metrics only logged during formal evaluations with `--eval_mode` flag

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    graph_mpe_runner.py                       │
│  - Handles training, debugging, informal testing             │
│  - Computes metrics (always)                                 │
│  - Does NOT save metrics (during debugging)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ --eval_mode flag
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              eval_metrics_logger.py (OPTIONAL)               │
│  - Only activated when --eval_mode=True                      │
│  - Zero overhead when disabled                               │
│  - Saves metrics to eval_summary.csv                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Formal Evaluation Pipeline                      │
│                                                               │
│  run_comprehensive_eval.py → eval_mpe.py --eval_mode         │
│         ↓                                                     │
│  compute_metrics_simple.py (reads eval_summary.csv)          │
│         ↓                                                     │
│  plot_results.py (creates figures)                           │
└─────────────────────────────────────────────────────────────┘
```

## File Responsibilities

### 1. `graph_mpe_runner.py` (EXISTING - NO CHANGES NEEDED)
**Purpose**: Training and debugging
**Metrics**: Computes but doesn't save (CSV writing commented out)
**Used for**: Day-to-day development

### 2. `eval_metrics_logger.py` (NEW - OPTIONAL MODULE)
**Purpose**: Metrics logging for formal evaluations only
**Activation**: Only when `--eval_mode` flag is set
**Output**: `eval_summary.csv` per evaluation run
**Overhead**: Zero when disabled (all methods are no-ops)

**Integration** (add to `graph_mpe_runner.py` render method):
```python
from .eval_metrics_logger import create_logger

# At start of render()
metrics_logger = create_logger(self.all_args)
metrics_logger.set_config(self.num_agents, self.all_args.world_size,
                         self.episode_length)

# In episode loop
metrics_logger.log_episode(
    conformance_pct=np.mean(conformance_percentage),
    success_rate=success * 100,
    completion_time=time_taken,
    delta_d=np.mean(delta_space),
    spacing_violations=np.mean(spacing_violations) * 100
)

# After all episodes
metrics_logger.save_summary()
```

### 3. `run_comprehensive_eval.py` (ORCHESTRATION)
**Purpose**: Run multiple scenarios × agent counts
**Key feature**: Automatically passes `--eval_mode` flag
**Output**: Organized directory structure

### 4. `compute_metrics_simple.py` (ANALYSIS)
**Purpose**: Aggregate metrics from eval_summary.csv files
**Input**: Directory with evaluation results
**Output**:
- `raw_metrics.csv` - All data
- `metrics_table.csv` - Summary table
- `metrics_table.tex` - LaTeX table for paper

### 5. `plot_results.py` (VISUALIZATION)
**Purpose**: Generate publication-quality figures
**Input**: `raw_metrics.csv`
**Output**: PDF figures

## Usage Patterns

### During Development (Debugging)
```bash
# Normal debugging - NO metrics saved
python onpolicy/scripts/eval_mpe.py \
    --model_dir=... \
    --scenario_name=simple_combined_graph \
    --num_agents=5 \
    --render_episodes=10
```
**Result**: Runs normally, no CSV pollution ✅

### Formal Evaluation (For Paper)
```bash
# Option 1: Automated suite
./eval_scripts/run_full_evaluation_suite.sh

# Option 2: Manual with --eval_mode
python onpolicy/scripts/eval_mpe.py \
    --model_dir=... \
    --scenario_name=simple_combined_graph \
    --num_agents=40 \
    --render_episodes=100 \
    --eval_mode \
    --eval_output_dir=eval_results/my_run
```
**Result**: Creates `eval_results/my_run/eval_summary.csv` ✅

### Analysis
```bash
python eval_scripts/compute_metrics_simple.py \
    --results_dir eval_results

python eval_scripts/plot_results.py \
    --metrics_file eval_results/raw_metrics.csv \
    --output_dir plots
```

## Metrics Computed

All metrics are computed in `graph_mpe_runner.py` using existing methods:

| Metric | Source Method | CSV Column |
|--------|---------------|------------|
| **C%** - Conformance | `get_conformation_percentages()` | conformance_pct_mean |
| **S%** - Success rate | `get_fraction_episodes()` | success_rate_mean |
| **T** - Completion time | `time_taken` | completion_time_mean |
| **Δd** - Separation violation | `get_delta_spacing()` | delta_d_mean |
| **I%** - Intervention need | `get_spacing_violations()` | spacing_violations_mean |

## Benefits of This Architecture

### ✅ Clean Separation
- **Debugging**: No metrics saved, no CSV pollution
- **Formal evals**: Metrics saved only when requested

### ✅ Zero Overhead
- `eval_metrics_logger` has zero cost when disabled
- All methods become no-ops if `enabled=False`

### ✅ No Code Duplication
- Metrics computation stays in `graph_mpe_runner.py`
- Logger just saves what's already computed

### ✅ Backward Compatible
- Existing code works unchanged
- CSV writing in `graph_mpe_runner.py` stays commented out
- Optional integration via single flag

### ✅ Easy to Use
- Single flag: `--eval_mode`
- Automated: `run_full_evaluation_suite.sh`
- Clear output structure

## Migration from Old System

If you have old CSV-writing code in `graph_mpe_runner.py`:

1. **Keep it commented out** (lines 1128-1134)
2. **Add eval_metrics_logger integration** (5 lines of code)
3. **Use --eval_mode flag** for formal evaluations
4. **Keep debugging as-is** (no flags, no CSV files)

## File Tree

```
AAM-Corridor-MARL/
├── onpolicy/
│   ├── runner/shared/
│   │   ├── graph_mpe_runner.py      (existing, minimal changes)
│   │   └── eval_metrics_logger.py   (new, optional module)
│   └── scripts/
│       └── eval_mpe.py               (existing, add --eval_mode arg)
├── eval_scripts/
│   ├── run_comprehensive_eval.py    (orchestration)
│   ├── compute_metrics_simple.py    (analysis)
│   ├── plot_results.py              (visualization)
│   ├── run_full_evaluation_suite.sh (one-command execution)
│   ├── ARCHITECTURE.md              (this file)
│   └── README_COMPREHENSIVE_EVAL.md (user guide)
└── eval_results/                    (created during eval)
    ├── scenario_agents10_seed1/
    │   └── eval_summary.csv
    ├── scenario_agents20_seed1/
    │   └── eval_summary.csv
    ├── raw_metrics.csv
    ├── metrics_table.csv
    └── metrics_table.tex
```

## FAQ

**Q: Do I need to modify graph_mpe_runner.py?**
A: Minimal changes - just add 5-10 lines to integrate eval_metrics_logger (optional).

**Q: Will this slow down my debugging?**
A: No! When --eval_mode is not set, the logger is disabled with zero overhead.

**Q: Can I still use the old CSV writing code?**
A: Yes, but it's not recommended since it mixes debugging and evaluation data.

**Q: What if I forget --eval_mode during evaluation?**
A: No metrics will be saved, compute_metrics_simple.py will warn you.

**Q: Can I use this with existing trained models?**
A: Yes! Just run evaluations with --eval_mode flag.
