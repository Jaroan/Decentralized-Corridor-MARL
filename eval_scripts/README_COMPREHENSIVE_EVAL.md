# Comprehensive AAM Corridor Evaluation Suite

This directory contains scripts for running comprehensive evaluations of AAM corridor scenarios and computing performance metrics for research paper submission.

## Overview

The evaluation suite consists of three main components:

1. **run_comprehensive_eval.py** - Orchestrates evaluation runs across multiple scenarios and agent counts
2. **compute_metrics.py** - Computes performance metrics from evaluation results
3. **metrics_tracker.py** - Helper module for tracking data during evaluations (imported by eval_mpe.py)

## Performance Metrics

The following metrics are computed, as defined for ATRDS2025 submission:

### 1. Conformance to Corridor Boundaries (C%)
- **Definition**: Average percentage of time aircraft stay within corridor boundaries while traversing
- **Range**: 0-100%
- **Better**: Higher values indicate better navigation performance

### 2. Success Rate (S%)
- **Definition**: Percentage of aircraft that successfully reach their assigned goals within episode length
- **Range**: 0-100%
- **Better**: Higher values indicate more efficient routes

### 3. Completion Time per Episode (T, seconds)
- **Definition**: Time for all aircraft to reach their goals (or episode timeout)
- **Better**: Lower values indicate more efficient navigation
- **Reported as**: Mean ± Std

### 4. Violation of Separation Minimum (Δd, meters)
- **Definition**: Amount by which 300m separation minimum is violated when violations occur
- **Example**: If two aircraft are 290m apart, Δd = 10m
- **Better**: Lower values indicate better separation maintenance
- **Reported as**: Mean ± Std (only for violation instances)

### 5. Need for Tactical Intervention (I%)
- **Definition**: Fraction of time when aircraft have separation violations relative to time in corridors
- **Range**: 0-100%
- **Better**: Lower values indicate less need for tactical deconfliction

## Scenarios

The suite evaluates five corridor scenarios:

1. **merge** - Simple merge scenario
2. **double_merge** - Double merge scenario
3. **split_merge** - Split and merge scenario
4. **sequential** - Multiple sequential corridors
5. **combined** - Complex combined network with "<" entry and C-shaped exit

## Agent Counts

Each scenario is tested with varying traffic densities:
- **10 agents** - Low density
- **20 agents** - Medium density
- **30 agents** - High density
- **40 agents** - Very high density

World size and episode length are automatically scaled based on agent count to maintain appropriate density and allow completion.

## Usage

### Step 1: Run Comprehensive Evaluations

```bash
# Run all scenarios with all agent counts (recommended)
python eval_scripts/run_comprehensive_eval.py --output_dir eval_results

# Run specific scenarios only
python eval_scripts/run_comprehensive_eval.py \
    --scenarios merge combined \
    --agent_counts 10 20 30 \
    --output_dir eval_results

# Dry run to see what will be executed
python eval_scripts/run_comprehensive_eval.py --dry_run

# Enable rendering (WARNING: very slow, only for visualization)
python eval_scripts/run_comprehensive_eval.py --render
```

**Important**: Each configuration runs 100 episodes. Expect ~10 minutes per configuration.
Total time for all scenarios: ~4-5 hours.

### Step 2: Compute Performance Metrics

After evaluations complete:

```bash
python eval_scripts/compute_metrics.py --results_dir eval_results
```

This will generate:
- `raw_metrics.csv` - All raw metric values per run
- `metrics_table.csv` - Aggregated results table
- `metrics_table.tex` - LaTeX-formatted table ready for paper

### Step 3: Verify Results

Check the summary JSON files for quick validation:

```bash
cat eval_results/merge_agents10_seed1/summary.json
```

## Integration with eval_mpe.py

To enable metrics tracking, you need to modify `onpolicy/scripts/eval_mpe.py`:

```python
# Add at the top
from metrics_tracker import MetricsTracker

# In the evaluation loop, initialize tracker
tracker = MetricsTracker(output_dir=args.eval_output_dir)

# In the episode loop
tracker.start_episode()

# In the timestep loop
tracker.record_step(world, scenario)

# After episode completes
tracker.end_episode(world, scenario, episode_time)

# After all episodes
tracker.save()
```

## Output Structure

```
eval_results/
├── merge_agents10_seed1/
│   ├── eval_data.pkl          # Tracked episode data
│   ├── summary.json           # Quick summary stats
│   ├── stdout.txt             # Evaluation stdout
│   └── stderr.txt             # Evaluation stderr
├── merge_agents20_seed1/
│   └── ...
├── combined_agents40_seed1/
│   └── ...
├── raw_metrics.csv            # All computed metrics
├── metrics_table.csv          # Aggregated table (CSV)
└── metrics_table.tex          # Aggregated table (LaTeX)
```

## Adding Heterogeneous Models

To test with multiple speed models (heterogeneity):

1. Train models with different max_speed values
2. Add to MODELS dict in run_comprehensive_eval.py:

```python
MODELS = {
    'standard': {
        'dir': 'model_weights/.../standard',
        'max_speed': 2.0,
    },
    'slow_v1': {
        'dir': 'model_weights/.../slow_v1',
        'max_speed': 1.5,
    },
    'slow_v2': {
        'dir': 'model_weights/.../slow_v2',
        'max_speed': 1.2,
    },
}
```

3. Run evaluations with different models:

```bash
python eval_scripts/run_comprehensive_eval.py --model slow_v1
python eval_scripts/run_comprehensive_eval.py --model slow_v2
```

## Customization

### Adjust World Size Scaling

Edit `get_world_size()` in run_comprehensive_eval.py:

```python
def get_world_size(base_size, num_agents):
    scale_factor = np.sqrt(num_agents / 10)  # Adjust this
    return int(base_size * scale_factor)
```

### Adjust Episode Length Scaling

Edit `get_episode_length()` in run_comprehensive_eval.py:

```python
def get_episode_length(base_length, num_agents):
    scale_factor = 1.0 + 0.1 * (num_agents - 10) / 10  # Adjust this
    return int(base_length * scale_factor)
```

### Add New Scenarios

Add to SCENARIOS dict in run_comprehensive_eval.py:

```python
SCENARIOS = {
    'my_scenario': {
        'name': 'scenario_file_name',
        'world_size_base': 8,
        'episode_length_base': 180,
    },
}
```

## Troubleshooting

### "eval_data.pkl not found"
- The eval_mpe.py script needs to be modified to use MetricsTracker
- See "Integration with eval_mpe.py" section above

### Evaluation runs out of memory
- Reduce the number of concurrent evaluations
- Disable rendering (--render flag)
- Reduce agent counts

### Metrics seem incorrect
- Check that separation_minimum matches your experiment (default: 300m)
- Verify corridor_width in compute_conformance() matches your setup
- Check that eval_mpe.py is correctly calling tracker.record_step()

## Paper Submission

For ATRDS2025 submission:

1. Run full evaluation suite (all scenarios, all agent counts)
2. Compute metrics
3. Use metrics_table.tex in your LaTeX document:

```latex
\input{eval_results/metrics_table.tex}
```

4. Discuss results in context of:
   - Separation assurance and safety nets
   - Advanced CNS for AAM
   - Autonomous systems integration into ATM
   - ATM performance measurement

## Citation

If using this evaluation framework, please cite the original work on AAM corridor navigation with MARL.

## Contact

For questions or issues, please contact the development team.
