#!/bin/bash
# Run split and merge corridor experiments with different agent counts
# Saves trajectories for average speed analysis

MODEL_DIR="model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/test"
SCENARIO="three_phase_graph_split_and_merge"
WORLD_SIZE=5
SEED=0

echo "Running split and merge corridor experiments..."
echo ""

# 10 agents - 50 episodes - episode length 250
echo "=== Experiment 1: 10 agents, 50 episodes ==="
python onpolicy/scripts/eval_mpe.py \
  --model_dir="$MODEL_DIR" \
  --scenario_name="$SCENARIO" \
  --num_agents=10 \
  --render_episodes=50 \
  --world_size=$WORLD_SIZE \
  --episode_length=200 \
  --seed=$SEED \
  --dynamics_type=air_taxi \
  --use_dones=False \
  --collaborative=False \
  --model_name='SplMer' \
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
  --eval_output_dir="eval_split_merge_10agents" \
  --save_trajectories

echo ""
echo "✓ 10 agents complete"
echo ""

# 20 agents - 50 episodes - episode length 300
echo "=== Experiment 2: 20 agents, 50 episodes ==="
python onpolicy/scripts/eval_mpe.py \
  --model_dir="$MODEL_DIR" \
  --scenario_name="$SCENARIO" \
  --num_agents=20 \
  --render_episodes=50 \
  --world_size=$WORLD_SIZE \
  --episode_length=350 \
  --seed=$SEED \
  --dynamics_type=air_taxi \
  --use_dones=False \
  --collaborative=False \
  --model_name='SplMer' \
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
  --eval_output_dir="eval_split_merge_20agents" \
  --save_trajectories

echo ""
echo "✓ 20 agents complete"
echo ""

# 30 agents - 50 episodes - episode length 400
echo "=== Experiment 3: 30 agents, 50 episodes ==="
python onpolicy/scripts/eval_mpe.py \
  --model_dir="$MODEL_DIR" \
  --scenario_name="$SCENARIO" \
  --num_agents=30 \
  --render_episodes=50 \
  --world_size=$WORLD_SIZE \
  --episode_length=450 \
  --seed=$SEED \
  --dynamics_type=air_taxi \
  --use_dones=False \
  --collaborative=False \
  --model_name='SplMer' \
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
  --eval_output_dir="eval_split_merge_30agents" \
  --save_trajectories

echo ""
echo "✓ 30 agents complete"
echo ""

# 40 agents - 10 episodes - episode length 550
echo "=== Experiment 4: 40 agents, 10 episodes ==="
python onpolicy/scripts/eval_mpe.py \
  --model_dir="$MODEL_DIR" \
  --scenario_name="$SCENARIO" \
  --num_agents=40 \
  --render_episodes=10 \
  --world_size=$WORLD_SIZE \
  --episode_length=550 \
  --seed=$SEED \
  --dynamics_type=air_taxi \
  --use_dones=False \
  --collaborative=False \
  --model_name='SplMer' \
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
  --eval_output_dir="eval_split_merge_40agents" \
  --save_trajectories

echo ""
echo "✓ 40 agents complete"
echo ""

echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"
echo ""
echo "Next step: Compute average speeds"
echo "  python eval_scripts/compute_split_merge_average_speeds.py"
