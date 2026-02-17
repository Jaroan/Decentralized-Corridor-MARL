#!/bin/bash
# Run 40-agent evaluation for simple_combined_graph scenario with trajectory saving
# This will generate data needed for outflow rate analysis

MODEL_DIR="model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/"
SCENARIO="simple_combined_graph"
NUM_AGENTS=40
WORLD_SIZE=8
EPISODE_LENGTH=450
NUM_EPISODES=5
OUTPUT_DIR="eval_40agents_combined_graph"

echo "Running 40-agent evaluation for simple_combined_graph scenario..."
echo "Model: $MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

python onpolicy/scripts/eval_mpe.py \
  --model_dir="$MODEL_DIR" \
  --scenario_name="$SCENARIO" \
  --num_agents=$NUM_AGENTS \
  --render_episodes=$NUM_EPISODES \
  --world_size=$WORLD_SIZE \
  --episode_length=$EPISODE_LENGTH \
  --seed=0 \
  --dynamics_type=air_taxi \
  --use_dones=False \
  --collaborative=False \
  --goal_rew=20 \
  --collision_rew=20 \
  --formation_rew=10 \
  --num_walls=0 \
  --eval_mode \
  --eval_output_dir="$OUTPUT_DIR" \
  --save_trajectories

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. View metrics:"
    echo "     cat $OUTPUT_DIR/eval_summary.csv"
    echo ""
    echo "  2. Plot outflow rate for each episode:"
    for i in $(seq 0 $(($NUM_EPISODES - 1))); do
        echo "     python eval_scripts/plot_outflow_rate.py \\"
        echo "       --trajectory_file $OUTPUT_DIR/trajectory_${NUM_AGENTS}agents_episode_${i}.npz \\"
        echo "       --output_dir $OUTPUT_DIR/plots"
    done
    echo ""
    echo "  3. Plot all trajectories:"
    echo "     python eval_scripts/plot_trajectories.py \\"
    echo "       --trajectory_file $OUTPUT_DIR/trajectory_${NUM_AGENTS}agents_episode_0.npz"
else
    echo ""
    echo "✗ Evaluation failed!"
fi
