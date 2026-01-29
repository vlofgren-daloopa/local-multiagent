# SWE-bench Prompt Sweep Analysis

## Overview
Analysis of prompt variations for SWE-bench Lite (verified with Hunameval-validated agents).

## Summary Table

| Version | Model    | Temp    | Agents | Valid Rate | Gen Rate   | Notes                                           |
| :------ | :------- | :------ | :----- | :--------- | :--------- | :---------------------------------------------- |
| **v2**  | **1.5b** | **0.2** | **11** | **58.0%**  | **100.0%** | Best performer. Few-shot with bug (ironically). |
| v14     | unknown  | 0.0     | 5      | 48.0%      | 100.0%     | V4 variant with inline examples.                |
| v3      | unknown  | 0.0     | 4      | 36.0%      | 100.0%     | Fixed few-shot.                                 |
| v11     | 7b       | 0.1     | 7      | 34.0%      | 100.0%     |                                                 |
| v13     | unknown  | 0.0     | 6      | 28.0%      | 100.0%     |                                                 |
| v4      | unknown  | 0.0     | 6      | 26.0%      | 100.0%     |                                                 |
| v12     | unknown  | 0.0     | 5      | 24.0%      | 100.0%     |                                                 |
| v15     | unknown  | 0.0     | 4      | 22.0%      | 100.0%     |                                                 |
| v6      | unknown  | 0.0     | 5      | 10.0%      | 100.0%     |                                                 |
| v8      | unknown  | 0.0     | 3      | 10.0%      | 100.0%     |                                                 |
| v10     | 7b       | 0.1     | 7      | 8.0%       | 100.0%     | Role-playing expert.                            |
| v7      | unknown  | 0.0     | 6      | 8.0%       | 100.0%     |                                                 |
| v5      | unknown  | 0.0     | 6      | 6.0%       | 100.0%     |                                                 |
| v9      | unknown  | 0.0     | 4      | 0.0%       | 100.0%     |                                                 |

## Key Findings

1.  **Winner: v2 (1.5B)** significantly outperforms others with **58%** validation rate.
2.  **7B Performance:** v11 (7B) reached 34%, but v10 (7B) struggled at 8%.
3.  **Missing Metadata:** Many runs (v3-v9, v12-v15) are missing explicit configuration in their logs/results, labeled as "unknown" model/temp.
4.  **Overwrite Behavior:**
    - The experiment script **appends** to `results.json` but **skips** already processed IDs.
    - If you run Model A, then Model B in the same directory:
        - `run.log` (shell output) is **overwritten** (shows Model B).
        - `results.json` retains Model A's results and **skips** Model B (because IDs are done).
    - This creates a mismatch where logs show Model B but results are actually Model A.

## Recommendations
- **Stick with v2 prompt** for now given the high success rate.
- **Isolate runs:** Always use distinct directories for different models/configs to avoid log/result mismatches.
- **Rerun v3-v9** if precise attribution is needed, ensuring logs are captured.
