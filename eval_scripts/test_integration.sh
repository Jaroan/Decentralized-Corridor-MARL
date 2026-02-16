#!/bin/bash
# Quick test to verify eval_metrics_logger integration works

echo "=========================================="
echo "Testing Evaluation Metrics Logger"
echo "=========================================="
echo ""

# Test 1: Run WITHOUT --eval_mode (should not create CSV)
echo "Test 1: Running WITHOUT --eval_mode (debugging mode)"
echo "Expected: No eval_summary.csv created"
echo ""

python onpolicy/scripts/eval_mpe.py \
    --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/test' \
    --scenario_name='simple_combined_graph' \
    --num_agents=5 \
    --render_episodes=2 \
    --world_size=8 \
    --episode_length=150 \
    --dynamics_type='air_taxi' \
    --eval_output_dir='test_output_debug' \
    2>&1 | grep -E "(Evaluation metrics|eval_summary)"

if [ -f "test_output_debug/eval_summary.csv" ]; then
    echo "❌ FAILED: CSV should NOT be created in debug mode"
else
    echo "✅ PASSED: No CSV created (as expected)"
fi

echo ""
echo "------------------------------------------"
echo ""

# Test 2: Run WITH --eval_mode (should create CSV)
echo "Test 2: Running WITH --eval_mode (evaluation mode)"
echo "Expected: eval_summary.csv created"
echo ""

python onpolicy/scripts/eval_mpe.py \
    --model_dir='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/test' \
    --scenario_name='simple_combined_graph' \
    --num_agents=5 \
    --render_episodes=2 \
    --world_size=8 \
    --episode_length=150 \
    --dynamics_type='air_taxi' \
    --eval_mode \
    --eval_output_dir='test_output_eval' \
    2>&1 | grep -E "(Evaluation metrics|eval_summary|✓)"

if [ -f "test_output_eval/eval_summary.csv" ]; then
    echo "✅ PASSED: CSV created successfully"
    echo ""
    echo "Contents of eval_summary.csv:"
    head -2 test_output_eval/eval_summary.csv
else
    echo "❌ FAILED: CSV should be created in eval mode"
fi

echo ""
echo "=========================================="
echo "Integration Test Complete"
echo "=========================================="
echo ""
echo "Cleaning up test outputs..."
rm -rf test_output_debug test_output_eval
echo "Done!"
