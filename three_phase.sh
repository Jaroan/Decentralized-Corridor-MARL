#!/bin/bash
# Slurm sbatch options
#SBATCH --job-name 3rot_col25
#SBATCH -a 0-1
#SBATCH --gres=gpu:volta:1
##SBATCH --cpus-per-task=8
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 20 # cpus per task


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# module unload anaconda/2022a
# Loading the required module
source /etc/profile
# module load anaconda/2023a
module load conda/Python-ML-2025b-pytorch
# export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib:$LD_LIBRARY_PATH

logs_folder="out_fair_informarl3"
mkdir -p $logs_folder
# Run the script
seed_max=2
n_agents=3

# "double_integrator" or "unicycle_vehicle or air_taxi"
chosen_dynamics_type="air_taxi"
formation_type="point"

# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
fair_wts=(1)
fair_rews=(10 10)

args_fair_wt=()
args_fair_rew=()

# iterate through all combos and make a list
for i in ${!fair_wts[@]}; do
    for j in ${!fair_rews[@]}; do
        args_fair_wt+=(${fair_wts[$i]})
        args_fair_rew+=(${fair_rews[$j]})
    done
done

seeds=(0 1)
datetime_str=$(date '+%y%m%d_%H%M%S')

if [ "$chosen_dynamics_type" == "air_taxi" ]; then
    str_dynamics_type="at"
    world_size=4
    episode_length=75
    num_env_steps=4000000
elif [ "$chosen_dynamics_type" == "unicycle_vehicle" ]; then
    str_dynamics_type="uv"
    world_size=4
    episode_length=150
    num_env_steps=15000000
elif [ "$chosen_dynamics_type" == "double_integrator" ]; then
    str_dynamics_type="di"
    world_size=4
    episode_length=25
    num_env_steps=5000000
else
    echo "Error: Unsupported dynamics type '$chosen_dynamics_type'"
    exit 1  # Exit with a non-zero status to indicate an error
fi



echo "datetime_str: ${datetime_str}"
echo "dynamics_type: ${chosen_dynamics_type}"
echo "formation_type: ${formation_type}"



# for seed in `seq ${seed_max}`;
# do
# # seed=`expr ${seed} + 3`
# echo "seed: ${seed}"
# # execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "air_corridor_unicycle_${n_agents}" \
--model_dir "model_weights/tube/rot_inv/airtaxi/try/three" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
--experiment_name "${str_dynamics_type}_${datetime_str}_3_phase_with_coll20_spacing10rot_eplen${episode_length}" \
--scenario_name "three_phase_graph" \
--dynamics_type ${chosen_dynamics_type} \
--fair_wt ${args_fair_wt[$SLURM_ARRAY_TASK_ID]} \
--fair_rew ${args_fair_rew[$SLURM_ARRAY_TASK_ID]} \
--num_agents=${n_agents} \
--num_landmarks=${n_agents} \
--collision_rew 20 \
--formation_rew 10 \
--n_training_threads 1 --n_rollout_threads 64 \
--num_mini_batch 1 \
--episode_length ${episode_length} \
--total_actions 9 \
--num_env_steps ${num_env_steps} \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--use_dones "False" \
--collaborative "False" \
--goal_rew 20 \
--num_walls 0 \
--zeroshift 10 \
--world_size=${world_size} \
--graph_feat_type "relative" \
--increase_fairness "False" \
--auto_mini_batch_size --target_mini_batch_size 8192 \
--formation_type ${formation_type} \
&> $logs_folder/${str_dynamics_type}_${datetime_str}_3_phase_with_coll20_spacing10rot_eplen${episode_length}_${seeds[$SLURM_ARRAY_TASK_ID]}