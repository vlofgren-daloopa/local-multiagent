#!/usr/bin/env python3
"""
Analyze results from multiple SWE-bench sweep versions (prompt sweeps).
"""

import sys
import glob
import re
import json
from pathlib import Path
from typing import List, Dict, Any

# Add script directory to path to import analyze_swebench_sweep
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analyze_swebench_sweep import load_sweep_results, print_results_table
except ImportError:
    print("Error: Could not import analyze_swebench_sweep. Make sure it is in the scripts directory.")
    sys.exit(1)

def parse_run_log(log_path: Path) -> Dict[str, Any]:
    """Parse run.log to extract configuration."""
    config = {
        "model": "unknown",
        "temp": 0.0,
        "n_agents": 0,
        "prompt_version": "unknown"
    }

    if not log_path.exists():
        return config

    try:
        content = log_path.read_text()

        # Extract Prompt Version
        m_prompt = re.search(r"Prompt Version: (v\d+)", content)
        if m_prompt:
            config["prompt_version"] = m_prompt.group(1)

        # Extract Agent Count
        m_agents = re.search(r"Initialized (\d+) agent\(s\)", content)
        if m_agents:
            config["n_agents"] = int(m_agents.group(1))

        # Extract Temp
        m_temp = re.search(r"with temp=([\d\.]+)", content)
        if m_temp:
            config["temp"] = float(m_temp.group(1))

        # Extract Model
        # Log line: Loading model from models/qwen2.5-coder-7b-instruct-q4_k_m.gguf...
        m_model = re.search(r"Loading model from .*?qwen2\.5-coder-([\d\.]+[bB]).*?\.gguf", content, re.IGNORECASE)
        if m_model:
            config["model"] = m_model.group(1).lower()
        else:
            # Fallback for model name in log
            m_model_generic = re.search(r"Loading model from .*?models/([^/]+)\.gguf", content)
            if m_model_generic:
                model_fname = m_model_generic.group(1).lower()
                if "1.5b" in model_fname:
                    config["model"] = "1.5b"
                elif "7b" in model_fname:
                    config["model"] = "7b"
                elif "14b" in model_fname:
                    config["model"] = "14b"
                elif "32b" in model_fname:
                    config["model"] = "32b"

    except Exception as e:
        print(f"Error parsing log {log_path}: {e}")

    return config

def load_single_sweep_result(sweep_dir: Path) -> List[Dict]:
    """Load results for a single sweep directory (flat structure)."""
    log_path = sweep_dir / "run.log"
    results_json = sweep_dir / "experimental" / "results.json"

    if not results_json.exists():
        results_json = sweep_dir / "results.json"

    if not results_json.exists():
        return []

    config = parse_run_log(log_path)

    try:
        with open(results_json) as f:
            data = json.load(f)

        issues = data.get("results", [])

        # Compute metrics
        total = len(issues)
        generated = sum(1 for r in issues if r.get("generated", r.get("success", False)))
        valid_patches = sum(1 for r in issues if r.get("valid_patch", False))

        # Agent distribution
        agent_counts = {}
        for r in issues:
            aid = r.get("agent_id", 0)
            agent_counts[aid] = agent_counts.get(aid, 0) + 1

        # Avg output length
        output_lengths = [
            len(r.get("model_output", ""))
            for r in issues
            if r.get("model_output")
        ]
        avg_output_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0

        # Validation stats
        has_diff = sum(1 for r in issues if r.get("validation", {}).get("has_diff_header", False))
        has_hunks = sum(1 for r in issues if r.get("validation", {}).get("has_hunks", False))

        # Fallback for n_agents if not in config
        n_agents = config["n_agents"]
        if n_agents == 0 and issues:
            # Infer from max agent_id
            max_agent_id = max((r.get("agent_id", 0) for r in issues), default=0)
            n_agents = max_agent_id if max_agent_id > 0 else 1

        return [{
            "model": config["model"],
            "temp": config["temp"],
            "n_agents": n_agents,
            "total_issues": total,
            "generated": generated,
            "valid_patches": valid_patches,
            "has_diff_header": has_diff,
            "has_hunks": has_hunks,
            "generation_rate": generated / total if total > 0 else 0,
            "validation_rate": valid_patches / total if total > 0 else 0,
            "agent_distribution": agent_counts,
            "avg_output_length": avg_output_len,
            "path": str(sweep_dir),
            "prompt_version": config["prompt_version"]
        }]

    except Exception as e:
        print(f"Error loading results from {results_json}: {e}")
        return []

def analyze_all_sweeps():
    base_results_dir = Path("results")
    sweep_dirs = sorted(glob.glob(str(base_results_dir / "swebench_sweep_v*")))

    if not sweep_dirs:
        print("No sweep directories found matching 'results/swebench_sweep_v*'")
        return

    print(f"Found {len(sweep_dirs)} sweep directories.")

    summary_data = []

    for sweep_dir_str in sweep_dirs:
        sweep_dir = Path(sweep_dir_str)
        print(f"\n{'='*80}")
        print(f"Analyzing: {sweep_dir}")

        # Check if it's a flat structure (has run.log)
        if (sweep_dir / "run.log").exists():
            results = load_single_sweep_result(sweep_dir)
        else:
            # Assume nested structure
            results = load_sweep_results(sweep_dir)

        if not results:
            print(f"No results found in {sweep_dir}")
            continue

        print_results_table(results)

        # Aggregate best result for this sweep
        if results:
            best = max(results, key=lambda x: x.get("validation_rate", 0))

            # Use directory name as version if prompt_version not found
            version = best.get("prompt_version", sweep_dir.name)
            if version == "unknown":
                version = sweep_dir.name

            summary_data.append({
                "version": version,
                "best_model": best['model'],
                "best_temp": best['temp'],
                "best_n": best['n_agents'],
                "valid_rate": best.get('validation_rate', 0),
                "generated_rate": best.get("generation_rate", 0),
                "valid_patches": best.get('valid_patches', 0),
                "total": best['total_issues']
            })

    print(f"\n{'='*80}")
    print(f"PROMPT SWEEP SUMMARY (Sorted by Valid Rate)")
    print(f"{'='*80}")
    print(f"| {'Version':<20} | {'Model':<5} | {'Temp':<4} | {'N':<3} | {'Valid Rate':<10} | {'Gen Rate':<10} |")
    print(f"|{'-'*22}|{'-'*7}|{'-'*6}|{'-'*5}|{'-'*12}|{'-'*12}|")

    # Sort summary by validation rate descending
    summary_data_sorted = sorted(summary_data, key=lambda x: x["valid_rate"], reverse=True)

    for item in summary_data_sorted:
        print(f"| {item['version']:<20} | {item['best_model']:<5} | {item['best_temp']:<4.1f} | {item['best_n']:<3} | "
              f"{item['valid_rate']:<10.1%} | {item['generated_rate']:<10.1%} |")

if __name__ == "__main__":
    analyze_all_sweeps()
