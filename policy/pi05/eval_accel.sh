#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/packages/openpi-client/src:${SCRIPT_DIR}:${PYTHONPATH}"

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
test_num=${7:-100}

export PI0_MLP_REUSE=${PI0_MLP_REUSE:-1}
export PI0_MLP_REUSE_REL_THRESHOLD=${PI0_MLP_REUSE_REL_THRESHOLD:-0.02}
export PI0_MLP_REUSE_MIN_SKIP_RATIO=${PI0_MLP_REUSE_MIN_SKIP_RATIO:-0.0}
export PI0_MLP_REUSE_UPDATE_CACHE=${PI0_MLP_REUSE_UPDATE_CACHE:-1}

export PI05_DENOISE_KV_MODE=${PI05_DENOISE_KV_MODE:-layer_accumulate}
export PI05_DENOISE_KV_LAYERS_PER_STEP=${PI05_DENOISE_KV_LAYERS_PER_STEP:-2}
export PI05_DENOISE_KV_INITIAL_CURRENT_LAYERS=${PI05_DENOISE_KV_INITIAL_CURRENT_LAYERS:-0}
export PI05_DENOISE_KV_CUTOFF_STEP=${PI05_DENOISE_KV_CUTOFF_STEP:-0}
export PI05_TORCH_COMPILE=${PI05_TORCH_COMPILE:-0}
ckpt_setting=${model_name}_accel_${PI05_DENOISE_KV_MODE}
if [ "${PI05_TRACE_ENABLE:-0}" = "1" ]; then
    ckpt_setting=${ckpt_setting}_trace
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mpi05 accelerated benchmark, mode=${PI05_DENOISE_KV_MODE}, test_num=${test_num}\033[0m"
echo -e "\033[33mMLP reuse=${PI0_MLP_REUSE}, rel_threshold=${PI0_MLP_REUSE_REL_THRESHOLD}\033[0m"
echo -e "\033[33mtorch compile=${PI05_TORCH_COMPILE}\033[0m"

cd "${REPO_ROOT}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --test_num ${test_num}
