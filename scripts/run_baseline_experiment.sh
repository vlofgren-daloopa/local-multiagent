#!/bin/bash

# Script to run baseline experiments for Qwen2.5-Coder models on SWE-bench
# Condition: Baseline (Single Agent)
# Prompt: v1 (Simple)
# Temperature: 0.1

# Ensure models directory exists
if [ ! -d "models" ]; then
    echo "Error: models directory not found!"
    exit 1
fi

# Function to run experiment
run_baseline() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    OUTPUT_DIR="results/swebench/baseline_${MODEL_NAME}"

    echo "================================================================="
    echo "Running Baseline Experiment for ${MODEL_NAME}"
    echo "Model: ${MODEL_PATH}"
    echo "Output: ${OUTPUT_DIR}"
    echo "================================================================="

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "Error: Model file not found at ${MODEL_PATH}"
        echo "Please ensure the model is placed in the models/ directory."
        return 1
    fi

    python3 scripts/run_swebench_experiment.py \
        --condition baseline \
        --model-path "${MODEL_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --prompt-version v1 \
        --model-temp 0.1 \
        --n-issues 50  # Run all issues
}

# 1. Run for Qwen2.5-Coder-1.5B-Instruct
run_baseline "1.5B" "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

# 2. Run for Qwen2.5-Coder-7B-Instruct
run_baseline "7B" "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

echo "================================================================="
echo "All baseline experiments completed."
echo "================================================================="
