#!/bin/bash
# Run prompt sweeps for 1.5B model in PARALLEL (Max 3 jobs)
# Usage: ./scripts/run_1.5b_sweeps.sh

MODEL_PATH="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MAX_JOBS=3

# Set GPU library paths (Fix for running on CPU)
export LD_LIBRARY_PATH=/home/victor-lofgren/.pyenv/versions/3.12.9/envs/slm-swarm/lib/python3.12/site-packages/nvidia/cublas/lib:/home/victor-lofgren/.pyenv/versions/3.12.9/envs/slm-swarm/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/victor-lofgren/.pyenv/versions/3.12.9/envs/slm-swarm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Generate list of commands
echo "Generating commands..."
> commands.txt

# Run for versions v3 to v15 (skipping v10, v11 which are 7b exclusive/good)
# Actually, v10/v11 might have 1.5b runs missing, but user said "keep them".
# Let's run v3-9 and v12-15.
# If user wants v10/11 1.5b runs, we should add them. The previous analysis showed v10/11 mainly had 7b results or were mixed.
# To be safe and complete, I will run 1.5b for ALL target versions (v3-v15) except perhaps v10/11 if they are "7b exclusive".
# The user said "not v2... this is the structure the others should follow" implying v3+ should have both.
# I will include v3-v15. If v10/v11 folders exist, this will add 1.5b subfolders to them without touching existing 7b folders.

VERSIONS=({5..15})

for ver in "${VERSIONS[@]}"; do
    PROMPT="v${ver}"

    # Single configuration: temp=0.1, n_agents=7
    temp=0.1
    n=7

    temp_str=$(echo $temp | tr '.' '_')
    output_dir="results/swebench_sweep_${PROMPT}/1.5b/temp${temp_str}_n${n}"

    # Construct command
    cmd="python scripts/run_swebench_experiment.py --condition experimental --model-path $MODEL_PATH --n-agents $n --router-temp $temp --prompt-version $PROMPT --output-dir $output_dir --n-issues 50"
    echo "$cmd" >> commands.txt
done

echo "Starting parallel execution with $MAX_JOBS jobs..."
echo "Total commands: $(wc -l < commands.txt)"

# Use xargs to run in parallel
cat commands.txt | xargs -P $MAX_JOBS -I {} sh -c '{}'

rm commands.txt
echo "1.5B Sweep Complete!"
