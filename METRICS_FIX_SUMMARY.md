# Metrics Fix Summary

## Changes Made

### 1. Conformance Metric - Inverted to Show Actual Conformance

**Problem**: Conformance was counting violations, so 0% = perfect and high % = bad.

**Solution**: Inverted the metric so **higher % = better** (more conforming).

**Changes**:
- [graph_mpe_runner.py:810-820](onpolicy/runner/shared/graph_mpe_runner.py#L810): Added inversion logic
  ```python
  violation_pct = np.mean(conformance_percentage) if len(conformance_percentage) > 0 else 0.0
  actual_conformance_pct = 100.0 - (violation_pct * 100.0)  # Now higher % = better
  ```

**Interpretation**:
- **100%** = Perfect conformance (no corridor violations)
- **0%** = Always violating corridors
- **Expected values**: ≥85% for well-trained policies

---

### 2. Success Rate - Redefined for Sequential Corridors

**Problem**: Success was based on distance to goal landmark, which doesn't properly account for sequential corridor traversal.

**Solution**: Success now means **reaching Phase 2 of the last corridor**.

**Changes**:
1. [simple_combined_graph.py:951](multiagent/custom_scenarios/simple_combined_graph.py#L951): Added `On_last_corridor` field
   ```python
   'On_last_corridor': self.current_tube[agent.id] == self.agent_routes[agent.id][-1],
   ```

2. [base_runner.py:559-578](onpolicy/runner/shared/base_runner.py#L559): Updated success logic
   ```python
   # Success = reached Phase 2 of last corridor
   if phase_reached >= 2 and on_last_corridor:
       success.append(1)
   ```

**Interpretation**:
- **Phase 0**: Pre-corridor (approaching)
- **Phase 1**: In-corridor (traversing)
- **Phase 2**: Post-corridor (successfully exited)
- **Success**: Agent must reach Phase 2 **while on their last corridor**
- For single-corridor scenarios: Phase 2 alone = success (backward compatible)

---

## Validation

### Before Running Evaluations

Run the validation script to verify metrics are working correctly:

```bash
./eval_scripts/validate_baseline.sh
```

### Expected Results

With the fixes, you should see:

**Conformance (C%)**:
- ✅ **≥85%** for well-trained policies
- ✅ **90-98%** for simple scenarios (3-5 agents)
- ⚠️ **<70%** indicates agents frequently leaving corridors

**Success Rate (S%)**:
- ✅ **≥90%** for simple baseline scenarios
- ✅ **85-98%** for complex multi-corridor scenarios
- Based on completing full corridor sequence (not just reaching goal position)

### Validation Checks

The validation script automatically verifies:
1. ✓ Success rate ≥90% (simple scenario)
2. ✓ Conformance ≥85% (corridor following)
3. ✓ Throughput ≤ theoretical max
4. ✓ Throughput scales with agent count

---

## Impact on Results

### What Changed in CSV Output

**No change to CSV column names**, but values now mean:

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| `conformance_pct_mean` | 5.2% (violations) | 94.8% (conformance) |
| `success_rate_mean` | Goal distance-based | Last corridor Phase 2-based |

### Comparing Old vs New Results

**⚠️ Important**: Results from evaluations run **before this fix** are **NOT directly comparable** to new results:

1. **Conformance values are inverted**: Old 5% ≈ New 95%
2. **Success definition changed**: Old goal-based ≠ New phase-based

If you need to compare:
- Re-run old evaluations with the fixed code
- Or manually invert old conformance values: `new_C% = 100 - old_C%`

---

## Testing the Fixes

### Quick Test (1 episode)

```bash
python onpolicy/scripts/eval_mpe.py \
    --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/test' \
    --scenario_name='simple_combined_graph' \
    --num_agents=5 \
    --render_episodes=1 \
    --world_size=5 \
    --episode_length=150 \
    --dynamics_type='air_taxi' \
    --eval_mode \
    --eval_output_dir='test_metrics_fix'
```

Check `test_metrics_fix/eval_summary.csv`:
- Conformance should be **high** (≥85%)
- Success should reflect agents completing their full route

### Full Validation

```bash
# Run baseline validation (3, 5, 10 agents)
./eval_scripts/validate_baseline.sh

# Check results
cat validation_baseline/three_phase_graph_updated_5ag/eval_summary.csv
```

---

## Files Modified

1. **[onpolicy/runner/shared/graph_mpe_runner.py](onpolicy/runner/shared/graph_mpe_runner.py)**
   - Lines 810-820: Conformance inversion logic

2. **[onpolicy/runner/shared/base_runner.py](onpolicy/runner/shared/base_runner.py)**
   - Lines 559-578: Success based on Phase 2 of last corridor

3. **[multiagent/custom_scenarios/simple_combined_graph.py](multiagent/custom_scenarios/simple_combined_graph.py)**
   - Line 952: Added `On_last_corridor` field to agent info

4. **[eval_scripts/compute_metrics_simple.py](eval_scripts/compute_metrics_simple.py)**
   - Lines 200-208: Updated metric descriptions

---

## Next Steps

1. **Run validation**: `./eval_scripts/validate_baseline.sh`
2. **Check conformance** values are ≥85% (not ≤15%)
3. **Check success** reflects completing corridors (not just reaching goal)
4. **If validation passes**: Run full evaluation suite

---

## Summary

✅ **Conformance**: Now correctly shows 100% = perfect (inverted from violations)
✅ **Success**: Now based on completing last corridor (Phase 2), not goal distance
✅ **Backward compatible**: Single-corridor scenarios still work with Phase 2 alone
✅ **Documentation updated**: All guides reflect correct interpretation

**Key takeaway**: Higher conformance % is now better! 🎉
