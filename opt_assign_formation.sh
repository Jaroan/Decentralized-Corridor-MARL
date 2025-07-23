#!/bin/bash

# to train informarl (the graph version; aka our method)

# Slurm sbatch options
#SBATCH --job-name formationocc
#SBATCH -a 0-1
#SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
## SBATCH -c 48 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022a
export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib:$LD_LIBRARY_PATH

logs_folder="out_formation"
mkdir -p $logs_folder
# Run the script

seed_max=2

n_agents=3

# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")

seeds=(0 1)

# for seed in `seq ${seed_max}`;
# do
# # seed=`expr ${seed} + 3`
# echo "seed: ${seed}"
# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "formation_test_gnn_update_${n_agents}" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
--experiment_name "base_formation_new_gnn" \
--scenario_name "simple_graph_formation" \
--num_agents=${n_agents} \
--collision_rew 20 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 5000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_dones "False" \
--collaborative "False" \
--goal_rew 20 \
--num_walls 0 \
--auto_mini_batch_size --target_mini_batch_size 8192 \
&> $logs_folder/out_formation_new_gnn_${seeds[$SLURM_ARRAY_TASK_ID]}

# python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
# --project_name "fair_test_3" \
# --env_name "GraphMPE" \
# --algorithm_name "rmappo" \
# --seed 0 \
# --experiment_name "base_formation_collab" \
# --scenario_name "simple_graph_formation" \
# --num_agents=3 \
# --collision_rew 20 \
# --n_training_threads 1 --n_rollout_threads 2 \
# --num_mini_batch 1 \
# --episode_length 25 \
# --num_env_steps 20000 \
# --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
# --user_name "marl" \
# --use_cent_obs "False" \
# --graph_feat_type "relative" \
# --use_dones "False" \
# --collaborative "False" \
# --goal_rew 20 \
# --auto_mini_batch_size --target_mini_batch_size 16 --use_wandb