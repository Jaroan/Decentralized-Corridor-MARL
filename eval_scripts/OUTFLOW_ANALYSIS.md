# Outflow Rate Analysis for Simple Combined Graph

This guide explains how to track and plot the outflow rate (throughput) over time for the `simple_combined_graph` scenario with 40 agents.

## Overview

The `simple_combined_graph` scenario has:
- **18 corridors** (T0–T17)
- **8 route options** per agent
- **2 exit corridors**: T16 (left exit) and T17 (right exit)
- All routes end at either T16 or T17

The outflow rate analysis tracks:
1. When agents exit through each corridor
2. The instantaneous outflow rate over time
3. Cumulative exits over the episode

## Quick Start

### Step 1: Run Evaluation with Trajectory Saving

```bash
# Using the helper script (recommended)
./eval_scripts/run_40agent_eval.sh

# Or manually:
python onpolicy/scripts/eval_mpe.py \
  --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/' \
  --scenario_name='simple_combined_graph' \
  --num_agents=40 \
  --render_episodes=5 \
  --world_size=8 \
  --episode_length=450 \
  --seed=0 \
  --dynamics_type=air_taxi \
  --use_dones=False \
  --collaborative=False \
  --goal_rew=20 \
  --collision_rew=20 \
  --formation_rew=10 \
  --num_walls=0 \
  --eval_mode \
  --eval_output_dir='eval_40agents_combined_graph' \
  --save_trajectories
```

This will create trajectory files in `eval_40agents_combined_graph/`:
- `trajectory_40agents_episode_0.npz`
- `trajectory_40agents_episode_1.npz`
- etc.

### Step 2: Plot Outflow Rate

```bash
# Plot outflow rate for a specific episode
python eval_scripts/plot_outflow_rate.py \
  --trajectory_file eval_40agents_combined_graph/trajectory_40agents_episode_0.npz \
  --output_dir eval_40agents_combined_graph/plots
```

This creates two plots:

1. **Outflow Rate Plot** (`*_outflow_rate.png/pdf`):
   - Shows instantaneous outflow rate (agents/minute) vs time
   - Separate curves for T16 (left), T17 (right), and total
   - Expected behavior: starts at 0, rises to steady state, drops to 0

2. **Cumulative Exits Plot** (`*_cumulative_exits.png/pdf`):
   - Shows total number of exited agents vs time
   - S-shaped curve showing ramp-up, steady flow, and completion

### Step 3: Customize Analysis

```bash
# Adjust sliding window size for smoother/sharper rate curves
python eval_scripts/plot_outflow_rate.py \
  --trajectory_file eval_40agents_combined_graph/trajectory_40agents_episode_0.npz \
  --output_dir eval_40agents_combined_graph/plots \
  --window_size 30  # Default: 20 timesteps

# Adjust timestep duration if different from default
python eval_scripts/plot_outflow_rate.py \
  --trajectory_file eval_40agents_combined_graph/trajectory_40agents_episode_0.npz \
  --output_dir eval_40agents_combined_graph/plots \
  --dt 0.5  # Default: 0.5 seconds per timestep
```

## Understanding the Plots

### Outflow Rate Plot

The **outflow rate** is computed using a sliding window:
- Window size: 20 timesteps (default) = 10 seconds
- At each timestep, counts exits in the previous window
- Converts to agents/minute

Expected pattern for 40 agents:
1. **Start (0-50s)**: Rate ≈ 0 (agents entering/traversing corridors)
2. **Steady state (50-200s)**: Rate plateaus (consistent throughput)
3. **End (200-225s)**: Rate drops to 0 (last agents exiting)

### Key Metrics Shown

- **T16 exits**: Agents exiting through left exit corridor
- **T17 exits**: Agents exiting through right exit corridor
- **Total exits**: Sum of both corridors
- **Peak rate**: Maximum throughput achieved
- **Mean rate**: Average throughput during active period

## Output Files

Each analysis generates:
```
eval_40agents_combined_graph/
├── eval_summary.csv                    # Overall metrics
├── trajectory_40agents_episode_0.npz   # Trajectory data
├── trajectory_40agents_episode_1.npz
├── ...
└── plots/
    ├── trajectory_40agents_episode_0_outflow_rate.png
    ├── trajectory_40agents_episode_0_outflow_rate.pdf
    ├── trajectory_40agents_episode_0_cumulative_exits.png
    └── trajectory_40agents_episode_0_cumulative_exits.pdf
```

## Comparing Scenarios

To compare different test scenarios:

1. **Heterogeneous vs Homogeneous speeds**:
```bash
# Heterogeneous (mixed speeds)
python eval_scripts/eval_heterogeneous_speeds.py \
  --num_agents 40 \
  --slow_percentage 25 \
  --output_dir eval_40agents_hetero \
  --save_trajectories

# Homogeneous (all same speed)
./eval_scripts/run_40agent_eval.sh

# Plot both
python eval_scripts/plot_outflow_rate.py \
  --trajectory_file eval_40agents_hetero/trajectory_40agents_heterogeneous_episode_0.npz
python eval_scripts/plot_outflow_rate.py \
  --trajectory_file eval_40agents_combined_graph/trajectory_40agents_episode_0.npz
```

2. **Different agent counts**: Modify `--num_agents` in the evaluation command

3. **Different scenarios**: Change `--scenario_name` parameter

## Troubleshooting

### No exits detected
- Check that `episode_length` is long enough for agents to complete routes
- Verify agents are reaching phase 2 of corridors T16/T17
- Increase episode length if needed

### Low throughput
- Check for collisions or spacing violations in `eval_summary.csv`
- Review trajectory visualization for bottlenecks
- Verify corridor conformance rates

### Missing trajectory data
- Ensure `--save_trajectories` flag is used
- Check that `--eval_mode` flag is set
- Verify trajectory file exists in output directory

## Additional Analysis

### Trajectory Visualization
```bash
python eval_scripts/plot_trajectories.py \
  --trajectory_file eval_40agents_combined_graph/trajectory_40agents_episode_0.npz
```

### Velocity Heatmaps
The trajectory plots also show:
- Agent velocity color-coding (purple=fast, yellow=slow)
- Corridor/gap velocity heatmaps
- Approach zones before entry corridors

### Performance Metrics
```bash
# View summary statistics
cat eval_40agents_combined_graph/eval_summary.csv

# Compare across runs
python eval_scripts/compute_metrics_simple.py \
  --results_dir eval_40agents_combined_graph
```
