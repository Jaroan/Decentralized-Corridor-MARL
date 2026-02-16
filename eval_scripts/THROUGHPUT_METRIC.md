# Throughput Metric for AAM Corridor Evaluations

## Overview

**Throughput (Θ)** measures the rate at which aircraft successfully exit the final corridor, expressed in **agents per minute**.

This is a critical operational metric for AAM corridors, indicating the system's capacity to process traffic.

## Definition

```
Throughput (Θ) = Number of successful exits / Episode time (in minutes)
```

### Example
- **10 agents** successfully exit over **120 seconds** (2 minutes)
- **Θ = 10 / 2 = 5 agents/minute**

## Why Throughput Matters

### 1. **Capacity Planning**
- Determines maximum sustainable traffic flow
- Helps design corridor dimensions and spacing requirements
- Informs scheduling and slot allocation

### 2. **Operational Efficiency**
- Higher throughput = better airspace utilization
- Lower throughput may indicate bottlenecks or conflicts

### 3. **Safety vs. Efficiency Trade-off**
- Very high throughput may compromise safety (more conflicts)
- Very low throughput may be inefficient (underutilized corridors)
- Optimal throughput balances both

### 4. **Scalability Assessment**
- Shows how well the system scales with agent count
- Identifies saturation points

## Implementation

### How It's Computed

1. **Track exit times**: Record when each agent completes the final corridor
2. **Count successful exits**: Only count agents that successfully reached their goal
3. **Calculate rate**: Divide by total episode time (in minutes)

### Code Location

**Tracking** (`graph_mpe_runner.py`):
```python
# Initialize per episode
agent_exit_times = [-1.0] * self.num_agents

# Record exit time when agent completes
for agent_idx in range(self.num_agents):
    if dones[0][agent_idx] and agent_exit_times[agent_idx] < 0:
        agent_exit_times[agent_idx] = current_time

# Compute throughput
successful_exits = [t for t in agent_exit_times if t >= 0]
throughput = len(successful_exits) / (episode_time / 60.0)
```

**Analysis** (`throughput_calculator.py`):
- `compute_throughput()` - Basic throughput calculation
- `compute_peak_throughput()` - Sliding window peak throughput
- `compute_inter_exit_time_stats()` - Time between exits

## Interpretation

### Expected Values

| Scenario | Agent Count | Typical Throughput |
|----------|-------------|-------------------|
| Simple merge | 10 | 4-6 agents/min |
| Simple merge | 20 | 8-12 agents/min |
| Simple merge | 40 | 12-18 agents/min |
| Complex combined | 10 | 3-5 agents/min |
| Complex combined | 40 | 10-15 agents/min |

*Values are approximate and depend on corridor geometry, policy quality, and traffic patterns.*

### What High Throughput Means

✅ **Positive indicators:**
- Efficient corridor traversal
- Good coordination between agents
- Minimal delays or conflicts

⚠️ **Potential concerns:**
- May indicate insufficient separation
- Could correlate with higher collision risk
- Check Δd and I% metrics

### What Low Throughput Means

⚠️ **Potential issues:**
- Bottlenecks in corridor design
- Poor agent coordination
- Excessive caution leading to delays

✅ **Positive aspect:**
- May indicate safer, more conservative operation
- Better separation maintenance

## Relationship to Other Metrics

### Throughput vs. Success Rate
- **High Θ, High S%**: Optimal - fast and successful
- **High Θ, Low S%**: Red flag - agents rushing but failing
- **Low Θ, High S%**: Conservative - safe but slow
- **Low Θ, Low S%**: Problem - slow and unsuccessful

### Throughput vs. Intervention Need (I%)
- **High Θ, Low I%**: Ideal - fast with few conflicts
- **High Θ, High I%**: Concerning - speed causing conflicts
- **Low Θ, Low I%**: Safe but inefficient
- **Low Θ, High I%**: Poor - slow despite conflicts

### Throughput vs. Completion Time (T)
- Generally inversely related
- Lower T → Higher Θ (faster completion)
- But Θ also depends on number of successful exits

## Advanced Analysis

### Peak Throughput

The **maximum throughput** observed in any 60-second window:

```python
peak_throughput, peak_time = compute_peak_throughput(exit_times, window_size=60.0)
```

Useful for:
- Identifying burst capacity
- Finding temporal bottlenecks
- Analyzing time-varying traffic patterns

### Inter-Exit Time

Statistics on **time between consecutive exits**:

```python
stats = compute_inter_exit_time_stats(exit_times)
# Returns: mean, std, min, max inter-exit times
```

Useful for:
- Understanding exit spacing
- Detecting clustering vs. uniform distribution
- Minimum safe inter-exit time

## Paper Reporting

### In Results Table

```latex
\begin{tabular}{llcccccc}
\hline
Scenario & Agents & C\% & S\% & T (s) & Δd (m) & I\% & Θ (ag/min) \\
\hline
Merge    & 10     & 95.2 & 98.5 & 105 & 5.2 & 2.1 & 5.7 \\
Merge    & 40     & 92.1 & 95.3 & 145 & 8.3 & 4.5 & 16.5 \\
\hline
\end{tabular}
```

### In Discussion

> "The system achieved a peak throughput of 16.5 agents/minute for the 40-agent scenario, demonstrating good scalability while maintaining a 95.3% success rate. The relatively low intervention need (4.5%) suggests that this throughput level is sustainable without excessive tactical deconfliction."

## Validation

### Sanity Checks

1. **Throughput ≤ (num_agents / min_episode_time)**
   - Cannot exceed theoretical maximum

2. **Throughput × episode_time ≈ successful_agents**
   - Should match (approximately)

3. **Higher agent count → Higher absolute throughput**
   - But not necessarily higher normalized throughput

### Example Validation

```python
# Episode: 10 agents, 120s, 9 successful
theoretical_max = 10 / (120/60)  # = 5.0 agents/min
observed = 9 / (120/60)          # = 4.5 agents/min
assert observed <= theoretical_max  # ✓ Valid
```

## Future Extensions

### 1. **Normalized Throughput**
```python
normalized_throughput = throughput / num_agents
# Accounts for agent count differences
```

### 2. **Throughput Efficiency**
```python
efficiency = observed_throughput / theoretical_max_throughput
# Percentage of theoretical capacity utilized
```

### 3. **Time-Varying Throughput Plot**
```python
# Plot throughput over time (sliding window)
for t in time_windows:
    window_throughput = count_exits_in_window(t, t+60) / 1.0
    plot(t, window_throughput)
```

## Summary

**Throughput (Θ)** is now tracked as the 6th performance metric:

1. **C%** - Conformance ↑
2. **S%** - Success rate ↑
3. **T** - Completion time ↓
4. **Δd** - Separation violation ↓
5. **I%** - Intervention need ↓
6. **Θ** - Throughput ↑  ← **NEW**

✅ **Automatically computed** when `--eval_mode` flag is set
✅ **Included in CSV summaries** and LaTeX tables
✅ **Helper functions** available in `throughput_calculator.py`

---

For ATRDS2025 paper submission, throughput demonstrates:
- **ATM performance measurement** - quantitative capacity metric
- **Scalability** - how system handles increasing traffic
- **Efficiency** - operational throughput vs. safety trade-offs
