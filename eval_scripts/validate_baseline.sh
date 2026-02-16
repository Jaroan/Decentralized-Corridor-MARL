#!/bin/bash
# Validation script for baseline single-corridor scenario
# Tests metrics computation on simple, well-understood scenario

echo "=========================================="
echo "Baseline Validation: three_phase_graph"
echo "=========================================="
echo ""

# Configuration
MODEL_DIR="model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/test"
SCENARIO="three_phase_graph_updated"  # Simple single-corridor baseline
OUTPUT_DIR="validation_baseline"
EPISODES=1  # Small number for quick validation

# Test with different agent counts
AGENT_COUNTS=(3 5 10)

echo "This will run quick validation tests with:"
echo "  Scenario: $SCENARIO (single corridor baseline)"
echo "  Agent counts: ${AGENT_COUNTS[@]}"
echo "  Episodes per test: $EPISODES"
echo "  Output: $OUTPUT_DIR/"
echo ""

# Clean previous validation results
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing previous validation results..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# Run evaluations for each agent count
for NUM_AGENTS in "${AGENT_COUNTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing with $NUM_AGENTS agents"
    echo "=========================================="

    TEST_DIR="${OUTPUT_DIR}/${SCENARIO}_${NUM_AGENTS}ag"

    python onpolicy/scripts/eval_mpe.py \
        --model_dir="$MODEL_DIR" \
        --scenario_name="$SCENARIO" \
        --num_agents=$NUM_AGENTS \
        --render_episodes=$EPISODES \
        --world_size=5 \
        --episode_length=150 \
        --dynamics_type='air_taxi' \
        --goal_rew=20 \
        --collision_rew=20 \
        --formation_rew=10 \
        --num_walls=0 \
        --use_dones=False \
        --collaborative=False \
        --eval_mode \
        --eval_output_dir="$TEST_DIR" \
        --save_gifs \
        2>&1 | tee "${TEST_DIR}/eval_log.txt"

    if [ $? -eq 0 ]; then
        echo "✓ Test with $NUM_AGENTS agents completed"
    else
        echo "✗ Test with $NUM_AGENTS agents failed"
    fi
done

echo ""
echo "=========================================="
echo "Validation Results Summary"
echo "=========================================="
echo ""

# Analyze results
for NUM_AGENTS in "${AGENT_COUNTS[@]}"; do
    TEST_DIR="${OUTPUT_DIR}/${SCENARIO}_${NUM_AGENTS}ag"
    CSV_FILE="${TEST_DIR}/eval_summary.csv"

    if [ -f "$CSV_FILE" ]; then
        echo "--- $NUM_AGENTS agents ---"

        # Extract key metrics using Python
        python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv('$CSV_FILE')

    print(f"  Episodes:        {df['num_episodes'].values[0]}")
    print(f"  Success Rate:    {df['success_rate_mean'].values[0]:.1f}%")
    print(f"  Conformance:     {df['conformance_pct_mean'].values[0]:.1f}%")
    print(f"  Completion Time: {df['completion_time_mean'].values[0]:.1f} ± {df['completion_time_std'].values[0]:.1f} s")
    print(f"  Throughput:      {df['throughput_mean'].values[0]:.2f} agents/min")
    print(f"  Δd (violations): {df['delta_d_mean'].values[0]:.2f} ± {df['delta_d_std'].values[0]:.2f} m")
    print(f"  Intervention:    {df['spacing_violations_mean'].values[0]:.1f}%")

    # Validation checks
    print("\n  Validation Checks:")

    # 1. Success rate should be high for simple scenario
    success = df['success_rate_mean'].values[0]
    if success >= 90:
        print(f"    ✓ Success rate is good ({success:.1f}% >= 90%)")
    else:
        print(f"    ⚠ Success rate is low ({success:.1f}% < 90%)")

    # 2. Throughput sanity check
    throughput = df['throughput_mean'].values[0]
    num_agents = $NUM_AGENTS
    episode_time = df['completion_time_mean'].values[0]
    theoretical_max = num_agents / (episode_time / 60.0)

    if throughput <= theoretical_max * 1.1:  # Allow 10% margin
        print(f"    ✓ Throughput is valid ({throughput:.2f} <= {theoretical_max:.2f} max)")
    else:
        print(f"    ✗ Throughput exceeds theoretical max ({throughput:.2f} > {theoretical_max:.2f})")

    # 3. Conformance should be high
    conformance = df['conformance_pct_mean'].values[0]
    if conformance >= 85:
        print(f"    ✓ Conformance is good ({conformance:.1f}% >= 85%)")
    else:
        print(f"    ⚠ Conformance is low ({conformance:.1f}% < 85%)")

    # 4. Check throughput scaling
    print(f"\n    Expected throughput scaling: ~{num_agents/3:.1f}x for {num_agents} agents vs 3 agents")

except Exception as e:
    print(f"  ✗ Error reading results: {e}", file=sys.stderr)
    sys.exit(1)
EOF

        echo ""
    else
        echo "--- $NUM_AGENTS agents ---"
        echo "  ✗ No results file found: $CSV_FILE"
        echo ""
    fi
done

# Create comparison table
echo "=========================================="
echo "Metric Comparison Table"
echo "=========================================="
echo ""

python3 << 'EOF'
import pandas as pd
import os

results = []
agent_counts = [3, 5, 10]

for num_agents in agent_counts:
    csv_file = f"validation_baseline/working_three_phase_graph_{num_agents}ag/eval_summary.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        results.append({
            'Agents': num_agents,
            'S%': df['success_rate_mean'].values[0],
            'C%': df['conformance_pct_mean'].values[0],
            'T(s)': df['completion_time_mean'].values[0],
            'Θ(ag/min)': df['throughput_mean'].values[0],
            'I%': df['spacing_violations_mean'].values[0],
        })

if results:
    comparison = pd.DataFrame(results)
    print(comparison.to_string(index=False))

    # Check scaling
    print("\nScaling Analysis:")
    if len(results) > 1:
        base_throughput = results[0]['Θ(ag/min)']
        base_agents = results[0]['Agents']

        for r in results[1:]:
            expected_ratio = r['Agents'] / base_agents
            actual_ratio = r['Θ(ag/min)'] / base_throughput
            scaling_efficiency = (actual_ratio / expected_ratio) * 100

            print(f"  {r['Agents']} agents: Throughput scaling {actual_ratio:.2f}x (expected {expected_ratio:.2f}x) = {scaling_efficiency:.1f}% efficiency")
else:
    print("No results to compare")
EOF

echo ""
echo "=========================================="
echo "Validation Complete"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review the validation checks above"
echo "  2. Verify throughput values make sense"
echo "  3. Check that metrics scale appropriately with agent count"
echo "  4. If all looks good, run full comprehensive evaluation"
echo ""
