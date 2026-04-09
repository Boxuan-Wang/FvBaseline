#!/bin/bash
# Sequential FV-guided continual training on
# ./data/cl_preprocessed_2/task_k/{train_mixed,val}/data.jsonl
# Usage:
#   SEED=42 MODEL_PATH=/path/to/model bash scripts/train/cl_preprocessed_seq_fvg.sh
# Optional env:
#   TASK_IDS="0 1 2 3 4" CUDA="0,1,2,3" EXP_NAME="cl_seq_fvg_42" FUNC_ROOT="./results/function_vector"
#   AUTO_COMPUTE_FV=1 FV_DATA_ROOT="./data/fv" FV_DATASET_TEMPLATE="task_{tid}/train/data.jsonl"
#   FV_DATASET_MAP="0:ni618_icl,1:ni1290_icl" FV_CUDA=0 FV_MODEL="llama-chat|/path/to/model"
set -euo pipefail
set -x

SEED="${SEED:-${1:-42}}"
CUDA="${CUDA:-4,5,6,7}"
PORT="${PORT:-29500}"
MODEL_PATH="${MODEL_PATH:-/home/admin/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B}"
CL_ROOT="${CL_ROOT:-/home/admin/workspace/aop_lab/collabmask/data/cl_preprocessed_2}"
TASK_IDS="${TASK_IDS:-0 1}"
EXP_NAME="${EXP_NAME:-cl_seq_fvg_${SEED}}"
FUNC_ROOT="${FUNC_ROOT:-./results/function_vector}"
AUTO_COMPUTE_FV="${AUTO_COMPUTE_FV:-1}"
FV_DATA_ROOT="${FV_DATA_ROOT:-./data/fv}"
FV_DATASET_TEMPLATE="${FV_DATASET_TEMPLATE:-task_{tid}/train/data.jsonl}"
FV_DATASET_MAP="${FV_DATASET_MAP:-}"
FV_EXP_NAME="${FV_EXP_NAME:-uni}"
FV_MAX_EVAL_SIZE="${FV_MAX_EVAL_SIZE:-100}"
FV_CUDA="${FV_CUDA:-${CUDA%%,*}}"
FV_MODEL="${FV_MODEL:-${MODEL_PATH}}"

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

resolve_fv_dataset_name() {
  local tid="$1"
  local template="${FV_DATASET_TEMPLATE}"
  local default_name=""

  # Auto-fix common typo like: task_{tid/train/data.jsonl}
  # Convert "<prefix>{tid/<suffix>}" -> "<prefix>{tid}/<suffix>"
  if [[ "${template}" =~ ^(.*)\{tid/(.*)\}$ ]]; then
    template="${BASH_REMATCH[1]}{tid}/${BASH_REMATCH[2]}"
    echo "Warning: corrected malformed FV_DATASET_TEMPLATE to '${template}'" >&2
  elif [[ "${template}" == *"{tid/"* ]]; then
    echo "Warning: FV_DATASET_TEMPLATE looks malformed: '${template}'" >&2
    echo "Expected placeholder format: '{tid}' (e.g. task_{tid}/train/data.jsonl)." >&2
  fi
  if [[ "${template}" != *"{tid}"* ]]; then
    echo "Warning: FV_DATASET_TEMPLATE does not contain '{tid}': '${template}'" >&2
  fi
  default_name="${template//\{tid\}/${tid}}"

  if [[ -n "${FV_DATASET_MAP}" ]]; then
    local entry key val
    for entry in ${FV_DATASET_MAP//,/ }; do
      key="${entry%%:*}"
      val="${entry#*:}"
      if [[ "${key}" == "${tid}" && -n "${val}" ]]; then
        echo "${val}"
        return 0
      fi
    done
  fi

  echo "${default_name}"
}

compute_function_vector_if_missing() {
  local tid="$1"
  local dataset_name="$2"
  local func_path="$3"
  local dataset_json=""

  if [[ -f "${func_path}" ]]; then
    return 0
  fi

  if [[ "${AUTO_COMPUTE_FV}" != "1" ]]; then
    echo "Function vector not found: ${func_path}"
    echo "Set FUNC_ROOT to precomputed vectors, or set AUTO_COMPUTE_FV=1."
    return 1
  fi

  if [[ "${dataset_name}" == /* ]]; then
    dataset_json="${dataset_name}"
  elif [[ "${dataset_name}" == *.json || "${dataset_name}" == *.jsonl ]]; then
    dataset_json="${FV_DATA_ROOT}/${dataset_name}"
  elif [[ -f "${FV_DATA_ROOT}/${dataset_name}.json" ]]; then
    dataset_json="${FV_DATA_ROOT}/${dataset_name}.json"
  elif [[ -f "${FV_DATA_ROOT}/${dataset_name}.jsonl" ]]; then
    dataset_json="${FV_DATA_ROOT}/${dataset_name}.jsonl"
  elif [[ -f "${FV_DATA_ROOT}/${dataset_name}/data.jsonl" ]]; then
    dataset_json="${FV_DATA_ROOT}/${dataset_name}/data.jsonl"
  else
    dataset_json="${FV_DATA_ROOT}/${dataset_name}"
  fi

  if [[ ! -f "${dataset_json}" ]]; then
    echo "Function-vector source dataset not found: ${dataset_json}"
    echo "Set FV_DATA_ROOT/FV_DATASET_TEMPLATE/FV_DATASET_MAP to a valid JSON/JSONL path."
    return 1
  fi

  echo "Function vector not found for task_${tid}; computing now from ${dataset_json} ..."
  fv_cmd="--universal_set --exp_name ${FV_EXP_NAME} --prefixes_type N --separators_type N --max_eval_size ${FV_MAX_EVAL_SIZE} --gen --no_eval --root_data_dir ${FV_DATA_ROOT}"
  bash scripts/eval_fv.sh "${FV_CUDA}" "${FV_MODEL}" "${dataset_name}" "${FUNC_ROOT}" "${fv_cmd}"

  if [[ ! -f "${func_path}" ]]; then
    echo "Function vector generation finished but expected file is still missing: ${func_path}"
    return 1
  fi
}

adapter_chain=""
stage=0
for tid in ${TASK_IDS}; do
  stage=$((stage + 1))
  task_subdir="task_${tid}"
  stage_name="${stage}${task_subdir}"
  output_dir="./results/${EXP_NAME}/${stage_name}"
  log_dir="./log/${EXP_NAME}/${stage_name}"
  mkdir -p "${output_dir}" "${log_dir}"

  fv_dataset_name="$(resolve_fv_dataset_name "${tid}")"
  func_path="${FUNC_ROOT}/${fv_dataset_name}/${FV_EXP_NAME}_function_vector.pt"
  compute_function_vector_if_missing "${tid}" "${fv_dataset_name}" "${func_path}"

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
