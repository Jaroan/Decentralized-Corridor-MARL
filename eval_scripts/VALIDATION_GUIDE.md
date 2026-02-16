# Validation Guide: Ensuring Metrics Make Sense

This guide helps you validate that the evaluation metrics are working correctly using a simple baseline scenario.

## Why Validate?

Before running comprehensive evaluations on complex scenarios, it's important to:
1. ✅ Verify metrics are computed correctly
2. ✅ Check that values are in reasonable ranges
3. ✅ Ensure throughput calculations make sense
4. ✅ Validate metric scaling with agent count

## Quick Validation

### Step 1: Run Baseline Test

```bash
./eval_scripts/validate_baseline.sh
```

This will:
- Test with 3, 5, and 10 agents
- Run 10 episodes each (quick)
- Use `working_three_phase_graph` (simple single corridor)
- Output results to `validation_baseline/`

**Expected runtime**: ~5-10 minutes total

### Step 2: Review Results

The script automatically checks:
- ✓ Success rate (should be ≥90% for simple scenario)
- ✓ Throughput validity (≤ theoretical maximum)
- ✓ Conformance (should be ≥85%)
- ✓ Throughput scaling (should increase with agent count)

## Manual Validation

### Test Single Scenario

```bash
python onpolicy/scripts/eval_mpe.py \
    --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/test' \
    --scenario_name='working_three_phase_graph' \
    --num_agents=5 \
    --render_episodes=10 \
    --world_size=6 \
    --episode_length=150 \
    --dynamics_type='air_taxi' \
    --eval_mode \
    --eval_output_dir='manual_validation'
```

### Check Results

```bash
# View summary
cat manual_validation/eval_summary.csv

# Or formatted
python3 << EOF
import pandas as pd
df = pd.read_csv('manual_validation/eval_summary.csv')
print("\nMetrics Summary:")
print("="*50)
for col in df.columns:
    if 'mean' in col:
        metric = col.replace('_mean', '')
        print(f"{metric:25s}: {df[col].values[0]:.2f}")
EOF
```

## Expected Values (Baseline Scenario)

### For `working_three_phase_graph` (single corridor)

| Agents | S% | C% | T(s) | Θ(ag/min) | I% | Δd(m) |
|--------|----|----|------|-----------|----|----|
| **3** | 95-100 | 90-98 | 80-120 | 1.5-2.5 | 0-5 | 0-10 |
| **5** | 90-100 | 88-96 | 90-130 | 2.5-4.0 | 1-8 | 2-15 |
| **10** | 85-98 | 85-94 | 100-150 | 4.0-7.0 | 2-12 | 5-20 |

*These are approximate ranges based on typical trained policies.*

### What to Look For

#### ✅ Good Signs

1. **Success Rate (S%)**:
   - High (>90%) for small agent counts (3-5)
   - Remains reasonable (>80%) for larger counts (10)
   - Indicates policy is working

2. **Conformance (C%)**:
   - High (>85%) across all agent counts
   - Shows agents follow corridor boundaries
   - Should stay relatively stable

3. **Throughput (Θ)**:
   - Increases with agent count (more agents → more exits)
   - Roughly scales linearly (10 agents ≈ 2-3× throughput of 3 agents)
   - Always ≤ theoretical max: `num_agents / (completion_time / 60)`

4. **Completion Time (T)**:
   - Increases slightly with agent count (more coordination needed)
   - Stays within episode length (shouldn't hit timeout often)

5. **Intervention Need (I%)**:
   - Low (<10%) for small agent counts
   - Increases gradually with more agents (more potential conflicts)

6. **Separation Violation (Δd)**:
   - Low values (<20m) indicate good separation
   - Some violations expected (agents learning to coordinate)

#### ⚠️ Warning Signs

1. **Very Low Success Rate (<70%)**:
   - Policy may not have learned the task
   - Check model weights are correct
   - Verify scenario matches training scenario

2. **Low Conformance (<70%)**:
   - Agents not following corridors
   - Check corridor parameters match training

3. **Throughput > Theoretical Max**:
   - Bug in exit time tracking
   - Check `agent_exit_times` are recorded correctly

4. **Throughput Doesn't Scale**:
   - Same throughput for 3 and 10 agents = bottleneck
   - May indicate corridor capacity limit

5. **Very High I% (>30%)**:
   - Excessive conflicts
   - May need better coordination or wider corridors

## Validation Checklist

Before running full evaluations, verify:

- [ ] Script runs without errors
- [ ] `eval_summary.csv` is created
- [ ] All 6 metrics are present in CSV
- [ ] Success rate is reasonable (>80%)
- [ ] Throughput ≤ theoretical maximum
- [ ] Throughput increases with agent count
- [ ] Conformance is high (>80%)
- [ ] No NaN or inf values in results

## Debugging Issues

### Issue: No `eval_summary.csv` created

**Cause**: `--eval_mode` flag not set

**Fix**:
```bash
# Make sure you include --eval_mode
python onpolicy/scripts/eval_mpe.py ... --eval_mode
```

### Issue: Throughput = 0.0

**Cause**: Exit times not being tracked

**Check**:
1. `agent_exit_times` initialized in episode loop
2. Exit times recorded when `dones[0][agent_idx]` is True
3. Throughput calculation includes successful exits

**Debug**:
```python
# Add print in graph_mpe_runner.py after line 813
print(f"Episode exit times: {agent_exit_times}")
print(f"Successful exits: {len([t for t in agent_exit_times if t >= 0])}")
print(f"Throughput: {throughput:.2f} ag/min")
```

### Issue: Throughput > Theoretical Max

**Cause**: Logic error in exit time tracking

**Check**:
```python
# Verify in graph_mpe_runner.py
episode_time = total_time_taken[-1]  # Should be actual episode time
successful_exits = [t for t in agent_exit_times if t >= 0]
theoretical_max = len(successful_exits) / (episode_time / 60.0)
```

### Issue: Metrics seem unrealistic

**Check**:
1. Scenario name matches training scenario
2. Model weights are loaded correctly
3. Episode length is appropriate
4. World size matches training configuration

## Advanced Validation

### Visualize Exit Times

```python
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you saved raw episode data
import pickle
with open('validation_baseline/.../eval_raw_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Plot throughput distribution
plt.figure(figsize=(10, 6))
plt.hist(data['throughput'], bins=20)
plt.xlabel('Throughput (agents/min)')
plt.ylabel('Frequency')
plt.title('Throughput Distribution Across Episodes')
plt.show()
```

### Compare Against Expected

```python
import pandas as pd

# Your results
results = pd.read_csv('validation_baseline/.../eval_summary.csv')

# Expected ranges (from table above)
expected = {
    'success_rate_mean': (90, 100),
    'conformance_pct_mean': (88, 96),
    'throughput_mean': (2.5, 4.0),  # for 5 agents
}

print("Validation against expected ranges:")
for metric, (low, high) in expected.items():
    value = results[metric].values[0]
    in_range = low <= value <= high
    status = "✓" if in_range else "⚠"
    print(f"{status} {metric}: {value:.2f} (expected {low}-{high})")
```

## After Validation

Once baseline validation passes:

1. **Run comprehensive evaluation**:
   ```bash
   ./eval_scripts/run_full_evaluation_suite.sh
   ```

2. **Compare complex vs. simple scenarios**:
   - Simple scenarios should have higher success rates
   - Complex scenarios may have lower throughput
   - Intervention rates should be higher in complex scenarios

3. **Validate scaling**:
   - Throughput should increase with agent count
   - But not perfectly linear (coordination overhead)

## Summary

**Quick validation workflow**:
```bash
# 1. Run baseline test
./eval_scripts/validate_baseline.sh

# 2. Check output
cat validation_baseline/working_three_phase_graph_5ag/eval_summary.csv

# 3. If all looks good, run full evaluation
./eval_scripts/run_full_evaluation_suite.sh
```

**Key metrics to verify**:
- ✅ Success rate: >80%
- ✅ Throughput: Reasonable and ≤ theoretical max
- ✅ Throughput scaling: Increases with agent count
- ✅ Conformance: >80%
- ✅ No NaN/inf values

---

**Once validation passes, you're ready for comprehensive evaluations!**
