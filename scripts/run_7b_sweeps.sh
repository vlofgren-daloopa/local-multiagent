#!/bin/bash
# Run prompt sweeps for 7B model SEQUENTIALLY
# Usage: ./scripts/run_7b_sweeps.sh

MODEL_PATH="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

# Set GPU library paths (Fix for running on CPU)
export LD_LIBRARY_PATH=/home/victor-lofgren/.pyenv/versions/3.12.9/envs/slm-swarm/lib/python3.12/site-packages/nvidia/cublas/lib:/home/victor-lofgren/.pyenv/versions/3.12.9/envs/slm-swarm/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/victor-lofgren/.pyenv/versions/3.12.9/envs/slm-swarm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Run for versions v3 to v15 (some might interact with existing 1.5b runs but we use separate folders)
VERSIONS=({3..15})

for ver in "${VERSIONS[@]}"; do
    PROMPT="v${ver}"

    # Single configuration: temp=0.1, n_agents=7
    temp=0.1
    n=7

    temp_str=$(echo $temp | tr '.' '_')
    output_dir="results/swebench_sweep_${PROMPT}/7b/temp${temp_str}_n${n}"

    echo "Running [${PROMPT}] temp=${temp}, n=${n}..."

    python scripts/run_swebench_experiment.py \
        --condition experimental \
        --model-path "$MODEL_PATH" \
        --n-agents $n \
        --router-temp $temp \
        --prompt-version "$PROMPT" \
        --output-dir "$output_dir" \
        --n-issues 50
done

echo "7B Sweep Complete!"
