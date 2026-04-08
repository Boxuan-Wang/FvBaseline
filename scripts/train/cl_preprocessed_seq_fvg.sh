#!/bin/bash
# Sequential FV-guided continual training on ./data/cl_preprocessed/task_k/{train_mixed,val}
# Usage:
#   SEED=42 MODEL_PATH=/path/to/model bash scripts/train/cl_preprocessed_seq_fvg.sh
# Optional env:
#   TASK_IDS="0 1 2 3 4" CUDA="0,1,2,3" EXP_NAME="cl_seq_fvg_42" FUNC_ROOT="./results/function_vector"
set -euo pipefail
set -x

SEED="${SEED:-${1:-42}}"
CUDA="${CUDA:-4,5,6,7}"
PORT="${PORT:-29500}"
MODEL_PATH="${MODEL_PATH:-/home/admin/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B}"
CL_ROOT="${CL_ROOT:-/home/admin/workspace/aop_lab/collabmask/data/cl_preprocessed_1}"
TASK_IDS="${TASK_IDS:-0 1}"
EXP_NAME="${EXP_NAME:-cl_seq_fvg_${SEED}}"
FUNC_ROOT="${FUNC_ROOT:-./results/function_vector}"

# Training hyperparams (aligned with existing seq0_fvg defaults)
BS="${BS:-32}"
GAS="${GAS:-2}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-3}"
EDIT_LAYER="${EDIT_LAYER:-9}"
PR_ALPHA="${PR_ALPHA:-1.0}"
KL_ALPHA1="${KL_ALPHA1:-20.0}"
KL_ALPHA2="${KL_ALPHA2:-0.1}"
KL_ALPHA3="${KL_ALPHA3:-1.0}"
REGULAR_LAYER_NUM="${REGULAR_LAYER_NUM:-1}"

mkdir -p "./results/${EXP_NAME}" "./log/${EXP_NAME}"

adapter_chain=""
stage=0
for tid in ${TASK_IDS}; do
  stage=$((stage + 1))
  task_subdir="task_${tid}"
  stage_name="${stage}${task_subdir}"
  output_dir="./results/${EXP_NAME}/${stage_name}"
  log_dir="./log/${EXP_NAME}/${stage_name}"
  mkdir -p "${output_dir}" "${log_dir}"

  func_path="${FUNC_ROOT}/${task_subdir}_icl/uni_function_vector.pt"
  if [[ ! -f "${func_path}" ]]; then
    echo "Function vector not found: ${func_path}"
    echo "Set FUNC_ROOT or precompute vectors per task before running."
    exit 1
  fi

  extra_args=(
    --do_train
    --do_eval
    --cl_preprocessed_root "${CL_ROOT}"
    --cl_task_subdir "${task_subdir}"
    --cl_train_split train_mixed
    --cl_val_split val
    --algo naive
    --model_name_or_path "${MODEL_PATH}"
    --output_dir "${output_dir}"
    --overwrite_output_dir
    --per_device_train_batch_size "${BS}"
    --per_device_eval_batch_size 2
    --lora_target "q_proj,v_proj"
    --bf16
    --gradient_accumulation_steps "${GAS}"
    --learning_rate "${LR}"
    --num_train_epochs "${EPOCHS}"
    --deepspeed configs/stage2_llama.config
    --run_name "${EXP_NAME}/${stage_name}"
    --max_source_length 512
    --max_target_length 128
    --generation_max_length 128
    --lr_scheduler_type constant
    --warmup_ratio 0.1
    --logging_strategy steps
    --logging_steps 20
    --evaluation_strategy epoch
    --save_strategy no
    --seed "${SEED}"
    --pr_alpha "${PR_ALPHA}"
    --pr_loss_type ind
    --local_model llama
    --fv_pr
    --fv_kl
    --edit_layer "${EDIT_LAYER}"
    --kl_alpha1 "${KL_ALPHA1}"
    --kl_alpha2 "${KL_ALPHA2}"
    --func_path "${func_path}"
  )

  if [[ "${stage}" -gt 1 ]]; then
    extra_args+=(
      --create_new_adapter True
      --adapter_name_or_path "${adapter_chain}"
      --kl_alpha3 "${KL_ALPHA3}"
      --regular_layer_num "${REGULAR_LAYER_NUM}"
    )
  fi

  deepspeed --include "localhost:${CUDA}" --master_port "${PORT}" src/run.py \
    "${extra_args[@]}" \
    > "${log_dir}/train.log" 2>&1

  if [[ -z "${adapter_chain}" ]]; then
    adapter_chain="${output_dir}"
  else
    adapter_chain="${adapter_chain},${output_dir}"
  fi
done

echo "Finished sequential FV-guided CL run: ${EXP_NAME}"
echo "Final adapter chain: ${adapter_chain}"
