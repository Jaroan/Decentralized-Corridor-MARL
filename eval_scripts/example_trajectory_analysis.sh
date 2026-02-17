#!/bin/bash
# Example workflow for trajectory analysis and heterogeneous speed testing

echo "=========================================="
echo "Trajectory Analysis Example Workflow"
echo "=========================================="
echo ""

MODEL_DIR="model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/"
SCENARIO="simple_combined_graph"

# ==================== STEP 1: Baseline (All Fast Agents) ====================
echo "STEP 1: Running baseline evaluation (all agents at 175 knots)..."
echo ""

python onpolicy/scripts/eval_mpe.py \
    --model_dir="$MODEL_DIR" \
    --scenario_name="$SCENARIO" \
    --num_agents=20 \
    --render_episodes=3 \
    --world_size=5 \
    --episode_length=350 \
    --dynamics_type='air_taxi' \
    --use_dones=False \
    --collaborative=False \
    --eval_mode \
    --save_trajectories \
    --eval_output_dir='trajectory_baseline'

echo ""
echo "✓ Baseline evaluation complete"
echo ""

# ==================== STEP 2: Visualize Baseline ====================
echo "STEP 2: Visualizing baseline trajectories..."
echo ""

python eval_scripts/plot_trajectories.py \
    --trajectory_file trajectory_baseline/trajectory_episode_0.npz \
    --output_dir trajectory_baseline/plots/

echo ""
echo "✓ Baseline plots saved to trajectory_baseline/plots/"
echo ""

# ==================== STEP 3: Instructions for Heterogeneous Testing ====================
echo "=========================================="
echo "NEXT STEPS: Heterogeneous Speed Testing"
echo "=========================================="
echo ""
echo "To test heterogeneous speeds:"
echo ""
echo "1. Edit multiagent/custom_scenarios/simple_combined_graph.py"
echo "   - Go to line ~310 (heterogeneous speeds section)"
echo "   - Uncomment the code block and modify:"
echo ""
echo "     slow_agent_ids = list(range(5))  # First 5 agents are slow"
echo "     slow_speed = 110 * 0.514444 * 0.001  # 110 knots"
echo "     fast_speed = 175 * 0.514444 * 0.001  # 175 knots"
echo "     for i, agent in enumerate(world.agents):"
echo "         if i in slow_agent_ids:"
echo "             agent.max_speed = slow_speed"
echo "             agent.color = np.array([0.85, 0.15, 0.15])  # Red"
echo "         else:"
echo "             agent.max_speed = fast_speed"
echo ""
echo "2. Re-run the evaluation:"
echo ""
echo "   python onpolicy/scripts/eval_mpe.py \\"
echo "       --model_dir='$MODEL_DIR' \\"
echo "       --scenario_name='$SCENARIO' \\"
echo "       --num_agents=20 \\"
echo "       --save_trajectories \\"
echo "       --eval_output_dir='trajectory_heterogeneous'"
echo ""
echo "3. Visualize with highlighted slow agents:"
echo ""
echo "   python eval_scripts/plot_trajectories.py \\"
echo "       --trajectory_file trajectory_heterogeneous/trajectory_episode_0.npz \\"
echo "       --highlight_agents 0 1 2 3 4 \\"
echo "       --output_dir trajectory_heterogeneous/plots/"
echo ""
echo "4. Compare results:"
echo "   - Check throughput: cat trajectory_heterogeneous/eval_summary.csv"
echo "   - Compare velocity heatmaps (baseline vs heterogeneous)"
echo "   - Analyze how fast agents adapt to slow agents"
echo ""
echo "=========================================="
echo "See eval_scripts/TRAJECTORY_ANALYSIS_GUIDE.md for detailed instructions"
echo "=========================================="
