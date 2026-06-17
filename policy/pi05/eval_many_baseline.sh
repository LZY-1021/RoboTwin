#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

task_list_file=${1}
default_task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
default_test_num=${7:-100}

if [ -z "${task_list_file}" ] || [ -z "${default_task_config}" ] || [ -z "${train_config_name}" ] || [ -z "${model_name}" ] || [ -z "${seed}" ] || [ -z "${gpu_id}" ]; then
    echo "Usage: bash policy/pi05/eval_many_baseline.sh <task_list_file> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [test_num]"
    echo "Task list format: task_name [task_config] [test_num]. Blank lines and lines starting with # are ignored."
    exit 1
fi

if [ ! -f "${task_list_file}" ]; then
    echo "Task list file not found: ${task_list_file}"
    exit 1
fi

failed_tasks=()

while read -r line || [ -n "${line}" ]; do
    line="${line%$'\r'}"
    line="${line%%#*}"
    read -r task_name task_config test_num _ <<< "${line}"

    if [ -z "${task_name}" ]; then
        continue
    fi

    task_config=${task_config:-${default_task_config}}
    test_num=${test_num:-${default_test_num}}

    echo -e "\033[36m[baseline] task=${task_name}, task_config=${task_config}, test_num=${test_num}\033[0m"
    if ! bash "${SCRIPT_DIR}/eval_baseline.sh" "${task_name}" "${task_config}" "${train_config_name}" "${model_name}" "${seed}" "${gpu_id}" "${test_num}"; then
        failed_tasks+=("${task_name}")
        echo -e "\033[31m[baseline] failed: ${task_name}\033[0m"
    fi
done < "${task_list_file}"

if [ ${#failed_tasks[@]} -ne 0 ]; then
    echo -e "\033[31mBaseline benchmark finished with failed tasks: ${failed_tasks[*]}\033[0m"
    exit 1
fi

echo -e "\033[32mBaseline benchmark finished successfully.\033[0m"
