#!/bin/bash
# Run heterogeneous simple combined graph experiments (5 episodes)
# Saves trajectories for outflow rate analysis

MODEL_DIR="model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width"
SCENARIO="simple_combined_graph"
WORLD_SIZE=8
SEED=1
NUM_AGENTS=40
NUM_EPISODES=5
EPISODE_LENGTH=550

echo "Running heterogeneous simple combined graph experiments..."
echo "Configuration:"
echo "  Scenario: $SCENARIO"
echo "  Num agents: $NUM_AGENTS"
echo "  Num episodes: $NUM_EPISODES"
echo "  Episode length: $EPISODE_LENGTH"
echo "  World size: $WORLD_SIZE"
echo ""

python onpolicy/scripts/eval_mpe.py \
  --model_dir="$MODEL_DIR" \
  --scenario_name="$SCENARIO" \
  --num_agents=$NUM_AGENTS \
  --render_episodes=$NUM_EPISODES \
  --world_size=$WORLD_SIZE \
  --episode_length=$EPISODE_LENGTH \
  --seed=$SEED \
  --dynamics_type=air_taxi \
  --use_dones=False \
  --collaborative=False \
  --model_name='Combined' \
  --goal_rew=20 \
  --collision_rew=20 \
  --formation_rew=10 \
  --fair_rew=1 \
  --num_walls=0 \
  --zeroshift=5 \
  --min_obs_dist=0.5 \
  --total_actions=5 \
  --get_metrics=True \
  --formation_type='point' \
  --eval_mode \
  --eval_output_dir="eval_heterogeneous_combined_graph" \
  --save_trajectories

echo ""
echo "✓ Heterogeneous experiments complete"
echo ""
echo "Next step: Plot averaged outflow rate"
echo "  python eval_scripts/plot_outflow_rate_averaged.py \\"
echo "    --trajectory_dir eval_heterogeneous_combined_graph \\"
echo "    --num_episodes $NUM_EPISODES \\"
echo "    --output_dir eval_heterogeneous_combined_graph/plots"
