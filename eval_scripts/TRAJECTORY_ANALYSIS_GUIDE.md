# Trajectory Analysis Guide for Heterogeneous Speed Testing

This guide explains how to save agent trajectories, visualize them, and test heterogeneous speed scenarios.

## Quick Start

### 1. Run Evaluation with Trajectory Logging

```bash
python onpolicy/scripts/eval_mpe.py \
    --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/' \
    --scenario_name='simple_combined_graph' \
    --num_agents=10 \
    --render_episodes=5 \
    --world_size=5 \
    --episode_length=350 \
    --dynamics_type='air_taxi' \
    --eval_mode \
    --save_trajectories \
    --eval_output_dir='trajectory_analysis'
```

**New Flag**: `--save_trajectories` saves positions, velocities, headings, phases, and tube indices at each timestep.

**Output**: Creates `trajectory_episode_0.npz`, `trajectory_episode_1.npz`, etc.

---

### 2. Visualize Trajectories

```bash
python eval_scripts/plot_trajectories.py \
    --trajectory_file trajectory_analysis/trajectory_episode_0.npz \
    --highlight_agents 0 1 2
```

**Generates**:
- `trajectories_all.png` - All agent paths with velocity color-coding
- `velocity_heatmap.png` - Spatial distribution of velocities
- `velocity_timeseries.png` - Velocity over time for selected agents
- `phase_transitions.png` - Corridor phase transitions over time

**Options**:
- `--highlight_agents 0 1 2` - Highlight specific agents (useful for slow agents)
- `--plot_agents 0 1 2 3 4` - Which agents to show in time series plots
- `--output_dir custom_plots/` - Custom output directory

---

## Heterogeneous Speed Testing

### Method 1: Using Scenario Modification (Recommended for Simple Tests)

For testing a few slow agents, you can manually modify the scenario to set different speeds:

1. **Edit `simple_combined_graph.py`** to add per-agent speed setting in `reset_world()`:

```python
# In reset_world() after agents are created, add:
def reset_world(self, world, num_current_episode=0):
    # ... existing code ...

    # Set heterogeneous speeds
    slow_agent_ids = [0, 1, 2]  # Agents 0-2 are slow
    slow_speed = 110 * 0.514444 * 0.001  # 110 knots in km/s
    fast_speed = 175 * 0.514444 * 0.001  # 175 knots in km/s

    for i, agent in enumerate(world.agents):
        if i in slow_agent_ids:
            agent.max_speed = slow_speed
        else:
            agent.max_speed = fast_speed
```

2. **Run evaluation**:
```bash
python onpolicy/scripts/eval_mpe.py \
    --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/' \
    --scenario_name='simple_combined_graph' \
    --num_agents=10 \
    --save_trajectories \
    --eval_output_dir='heterogeneous_slow3'
```

3. **Visualize with highlighting**:
```bash
python eval_scripts/plot_trajectories.py \
    --trajectory_file heterogeneous_slow3/trajectory_episode_0.npz \
    --highlight_agents 0 1 2
```

---

### Method 2: Using Config File (For Systematic Studies)

For testing multiple configurations systematically:

1. **Create speed config file** (`speed_configs/config_25pct_slow.json`):
```json
{
    "fast_speed_knots": 175,
    "slow_speed_knots": 110,
    "slow_agent_percentage": 25,
    "description": "25% slow agents at 110 knots"
}
```

2. **Create batch evaluation script** (`eval_scripts/batch_heterogeneous.sh`):
```bash
#!/bin/bash

SCENARIO="simple_combined_graph"
MODEL_DIR="model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/"

# Test different slow percentages
for SLOW_PCT in 0 10 25 50; do
    OUTPUT_DIR="heterogeneous_results/slow_${SLOW_PCT}pct"

    echo "Testing ${SLOW_PCT}% slow agents..."

    # Modify scenario and run
    python eval_scripts/run_heterogeneous_eval.py \
        --scenario_name=$SCENARIO \
        --model_dir=$MODEL_DIR \
        --num_agents=40 \
        --slow_percentage=$SLOW_PCT \
        --slow_speed=140 \
        --fast_speed=175 \
        --save_trajectories \
        --output_dir=$OUTPUT_DIR
done
```

---

## Analysis Workflow for Congestion Study

### Scenario: Study how fast agents adapt to slow agents

**Setup**: 40 agents, agents 0-9 are slow (110 knots), agents 10-39 are fast (175 knots)

**Step 1: Run evaluation**
```bash
python onpolicy/scripts/eval_mpe.py \
    --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/' \
    --scenario_name='simple_combined_graph' \
    --num_agents=40 \
    --render_episodes=10 \
    --save_trajectories \
    --save_gifs \
    --eval_output_dir='congestion_study'
```

(After manually setting speeds as described above)

**Step 2: Visualize trajectories**
```bash
# Highlight the slow agents
python eval_scripts/plot_trajectories.py \
    --trajectory_file congestion_study/trajectory_episode_0.npz \
    --highlight_agents 0 1 2 3 4 5 6 7 8 9 \
    --output_dir congestion_study/plots/
```

**Step 3: Analyze congestion patterns**

Create custom analysis script (`analyze_congestion.py`):
```python
import numpy as np
import matplotlib.pyplot as plt

# Load trajectory data
data = np.load('congestion_study/trajectory_episode_0.npz')
positions = data['positions']  # (steps, agents, 2)
velocities = data['velocities']  # (steps, agents)

# Identify slow agents
slow_ids = list(range(10))
fast_ids = list(range(10, 40))

# Compute pairwise distances at each timestep
def compute_min_separations(positions, agent_group1, agent_group2):
    """Compute minimum separation between two groups over time."""
    min_seps = []
    for t in range(len(positions)):
        pos1 = positions[t, agent_group1, :]
        pos2 = positions[t, agent_group2, :]
        # Compute all pairwise distances
        dists = np.linalg.norm(pos1[:, None, :] - pos2[None, :, :], axis=2)
        min_seps.append(dists.min())
    return np.array(min_seps)

# Analyze separation between slow and fast agents
min_sep_slow_fast = compute_min_separations(positions, slow_ids, fast_ids)

# Plot minimum separation over time
plt.figure(figsize=(12, 6))
plt.plot(min_sep_slow_fast * 1000, linewidth=2)  # Convert km to m
plt.axhline(152.4, color='r', linestyle='--', label='Minimum separation (152.4m)')
plt.xlabel('Time step')
plt.ylabel('Minimum Separation (m)')
plt.title('Minimum Separation Between Slow and Fast Agents')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('congestion_study/plots/min_separation.png', dpi=300)
plt.close()

# Identify congestion events (separation < threshold)
threshold = 0.200  # km = 200m
congestion_steps = np.where(min_sep_slow_fast < threshold)[0]
print(f"Congestion events: {len(congestion_steps)} timesteps")
print(f"Congestion percentage: {100*len(congestion_steps)/len(min_sep_slow_fast):.1f}%")
```

**Step 4: Create velocity adaptation plots**
```python
# Compare velocity profiles of agents near slow agents vs far away
def identify_trailing_agents(positions, velocities, slow_ids, threshold_dist=0.5):
    """Identify fast agents that are trailing slow agents."""
    trailing = []
    for t in range(len(positions)):
        slow_pos = positions[t, slow_ids, :]
        for agent_id in range(10, 40):  # Fast agents
            agent_pos = positions[t, agent_id, :]
            dists_to_slow = np.linalg.norm(slow_pos - agent_pos, axis=1)
            if dists_to_slow.min() < threshold_dist:
                trailing.append((t, agent_id, velocities[t, agent_id]))
    return trailing

trailing_data = identify_trailing_agents(positions, velocities, slow_ids)
# Analyze velocity reduction when trailing slow agents...
```

---

## Understanding Trajectory Data

### Data Structure

The `.npz` file contains:

```python
{
    'positions': (num_steps, num_agents, 2),      # x, y positions in km
    'velocities': (num_steps, num_agents),        # speed in km/s
    'headings': (num_steps, num_agents),          # heading angle in radians
    'phases': (num_steps, num_agents),            # 0=pre, 1=in, 2=post corridor
    'tubes': (num_steps, num_agents),             # current tube index
    'dt': float,                                  # timestep duration (seconds)
    'num_agents': int,                            # number of agents
    'episode_length': int,                        # actual steps taken
    'agent_exit_times': (num_agents,)             # when each agent exited
}
```

### Loading and Using Data

```python
import numpy as np

# Load data
data = np.load('trajectory_episode_0.npz')

# Extract specific agent's trajectory
agent_id = 0
agent_x = data['positions'][:, agent_id, 0]
agent_y = data['positions'][:, agent_id, 1]
agent_v = data['velocities'][:, agent_id]

# Convert velocity to knots for display
agent_v_knots = agent_v / 0.514444 / 0.001

# Find when agent entered corridor
first_in_corridor = np.where(data['phases'][:, agent_id] == 1)[0]
if len(first_in_corridor) > 0:
    entry_time = first_in_corridor[0] * data['dt']
    print(f"Agent {agent_id} entered corridor at t={entry_time:.1f}s")
```

---

## Expected Results for Heterogeneous Speed Tests

### Baseline (All agents at 175 knots)
- **Throughput**: ~15-18 agents/min (for 40 agents)
- **Success Rate**: ~95%
- **Velocity heatmap**: Uniform high velocity throughout

### 25% Slow Agents (140 knots)
- **Throughput**: Slightly lower (~14-16 agents/min)
- **Congestion**: Possible bunching behind slow agents
- **Velocity heatmap**: Lower velocity regions where slow agents are
- **Adaptation**: Fast agents may slow down when trailing

### 50% Slow Agents (110 knots)
- **Throughput**: Significantly lower (~10-12 agents/min)
- **Congestion**: More frequent close encounters
- **Spacing violations**: May increase due to speed differentials
- **Velocity heatmap**: Clear bimodal distribution

---

## Visualization Gallery

### 1. Trajectory Plot
- **Color**: Velocity (blue=slow, yellow/green=fast)
- **Thickness**: Highlighted agents are thicker
- **Markers**: Start (square), End (triangle)
- **Use**: See overall flow patterns, identify bottlenecks

### 2. Velocity Heatmap
- **Color**: Red=slow, Green=fast
- **Use**: Identify congestion zones, corridor bottlenecks
- **Look for**: Red regions = congestion/slowdowns

### 3. Velocity Time Series
- **Shows**: How individual agent speeds change over time
- **Use**: See when agents slow down (adaptation to slow agents)
- **Look for**: Drops in velocity = yielding behavior

### 4. Phase Transitions
- **Shows**: When agents enter/exit corridors
- **Use**: Verify corridor traversal, identify agents stuck in Phase 0
- **Look for**: Long Phase 0 = waiting to enter, Phase 1→0 = corridor violations

---

## Troubleshooting

### Issue: No trajectory files created
**Solution**: Make sure you used `--save_trajectories` flag

### Issue: Visualization fails with "invalid array shape"
**Solution**: Check that trajectory file is complete (episode finished)

### Issue: Velocities all zero
**Solution**: Check that agent dynamics are correctly initialized

### Issue: Can't see heterogeneous speeds
**Solution**: Verify agent.max_speed was set in scenario reset_world()

---

## Summary Commands

```bash
# 1. Basic trajectory collection
python onpolicy/scripts/eval_mpe.py \
    --save_trajectories \
    --eval_output_dir=traj_output

# 2. Visualize
python eval_scripts/plot_trajectories.py \
    --trajectory_file traj_output/trajectory_episode_0.npz

# 3. Heterogeneous speed (after manual scenario modification)
python onpolicy/scripts/eval_mpe.py \
    --num_agents=40 \
    --save_trajectories \
    --highlight_agents 0 1 2 3 4  # Slow agents

# 4. Batch testing
./eval_scripts/batch_heterogeneous.sh
```

---

## Next Steps for Your Analysis

1. ✅ **Baseline test**: Run with all agents at 175 knots, save trajectories
2. ✅ **Heterogeneous test**: Modify scenario to set agents 0-9 at 110 knots
3. ✅ **Visualize**: Use plot_trajectories.py with --highlight_agents 0 1 2 3 4 5 6 7 8 9
4. ✅ **Analyze congestion**: Create custom analysis for pairwise distances
5. ✅ **Velocity adaptation**: Plot how fast agents slow down when near slow agents
6. ✅ **Compare**: Run multiple percentages (10%, 25%, 50% slow) and compare throughput

**Key questions to answer**:
- At what distance do fast agents start slowing down?
- How much do fast agents reduce speed when trailing?
- What percentage of slow agents causes significant congestion?
- Does congestion appear at specific corridor locations (merges, splits)?
