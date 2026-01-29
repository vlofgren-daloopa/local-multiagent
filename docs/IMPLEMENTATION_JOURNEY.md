# Implementation Journey: From Practical Swarm to Emergent Research

**Date:** January 27, 2026
**Status:** Living Document
**Author:** Generated from Repository History

---

## 1. Introduction

This document details the evolutionary arc of this repository, tracing its path from a practical engineering tool designed for local code generation into a rigorous scientific instrument for researching emergent behavior in Small Language Models (SLMs). The project has undergone a significant pivot, moving from explicit multi-agent architecture to a study of self-organizing systems.

---

## 2. Phase 1: The Original Agent Swarm
**Source:** `docs/plans/0_original_agent_swarm/2025-01-18-local-multi-agent-coding-system.md`

### 2.1 The Vision
The project began with a pragmatic goal: **build a swarm-adjacent multi-agent system using SLMs that can run on 8-16GB GPUs.** The intention was to democratize AI-assisted coding by proving that consumer hardware could support complex multi-agent workflows previously reserved for cloud-based LLMs.

### 2.2 Implementation Details
The initial architecture was a **Hybrid Swarm**, characterized by explicit role assignments:
*   **Coordinator Agent:** Managed task distribution and decomposition.
*   **Specialist Agents:** Hardcoded roles including:
    *   `CodeGenerator`: Wrote the implementation.
    *   `TestWriter`: Created `pytest` cases.
    *   `Reviewer`: Analyzed code quality and security.
    *   `Debugger`: Fixed implementation errors.

### 2.3 Technology Stack
*   **Language:** Python 3.9+
*   **Orchestration:** `asyncio` for concurrent agent operations.
*   **Models:** Designed for quantization (GGUF/ExLlama) to fit 8GB VRAM boundaries.
*   **Structure:** A classic `src/` layout with `agents/`, `coordination/`, and `models/` modules.

### 2.4 Outcome
This phase successfully established the infrastructure for local inference and agent communication. However, it relied heavily on "prompt engineering" to force models into specific roles (e.g., "You are a QA engineer...").

---

## 3. The Pivot: Emergent Role Specialization
**Source:** `docs/plans/pivot/research-plan.md`

### 3.1 The Paradigm Shift
The project pivoted from **engineering** (building a better coding tool) to **science** (understanding how agents learn). The core hypothesis shifted: *Can a population of identical, frozen SLMs develop functional specialization through task-driven experience alone, without explicit prompting?*

### 3.2 New Research Questions
*   **RQ1 (Emergence):** Can specialization emerge in ≤3B parameter models without weight updates?
*   **RQ2 (Mechanisms):** What feedback loops drive this differentiation?
*   **RQ3 (Scaling):** What is the optimal population size before coordination costs outweigh benefits?

### 3.3 Methodology: Context-Only Learning
The repository was refactored to support a new experimental design:
*   **Identical Agents:** All agents start as clones (same system prompt, same model weights).
*   **Differentiation Mechanism:** Agents diverge solely based on their `experience_buffer` (the history of tasks they have successfully solved).
*   **Affinity Routing:** Tasks are routed to agents based on their past success with similar problems, reinforcing specialization loops.

---

## 4. Phase 2: Validation & The Crisis
**Source:** `docs/POSTMORTEM_HUMANEVAL_BUG.md`

### 4.1 The "Too Good to Be True" Pilot
Initial pilot runs on the HumanEval benchmark showed an astonishing **96% Pass@1** rate for a 1.5B parameter model. This result defied all improved baselines (typically ~30-40% for models of this size).

### 4.2 The Bug (January 24, 2026)
A deep investigation revealed a critical flaw in the test execution harness.
*   **The Issue:** The wrapper function for executing HumanEval tests created a nested scope where assertions were defined but **never called**.
*   **The Code:**
    ```python
    def check(candidate):
        def check(candidate): # INNER (From benchmark)
            assert ...        # Never executed
    ```
*   **The Impact:** All tests "passed" because no code was actually run. This invalidated all data collected during the initial experimental window.

### 4.3 The Fix & Prevention
The execution harness was rewritten to properly invoke the test suite. Strict validation protocols were introduced:
*   Sanity checks with "known wrong" solutions (must fail).
*   Logging of combined code execution.
*   Comparison against published baselines to flag anomalies.

---

## 5. Current State: Scaling & Future Benchmarks
**Source:** `docs/plans/PHASE_2_EXECUTION.md` & `docs/thesis_consolidated_framework.md`

### 5.1 Rigorous Re-Validation
Following the postmortem, the immediate focus is on generating statistically significant, valid data:
*   **Benchmarks:** Full HumanEval (164 tasks) and MBPP (500 tasks).
*   **Experiments:**
    *   **Baseline:** Single Agent vs. 3-Agent Random Swarm.
    *   **Specialization:** 3-Agent Affinity Swarm (Testing the thesis hypothesis).
    *   **Scaling:** Testing population sizes N=5 and N=7.

### 5.2 SWE-bench Integration
The "Holy Grail" of the current roadmap is moving beyond function-level synthesis (HumanEval) to repository-level problem solving.
*   **Target:** SWE-bench Lite (300 real GitHub issues).
*   **Goal:** Demonstrate that emergent swarms can solve complex integration problems better than monolithic large models.
*   **Implementation:** Using the official SWE-bench harness to properly evaluate patch generation and test passing rates.

### 5.3 Thesis Alignment
The repository now serves as the primary artifact for the Master's Thesis: *"Emergent Role Specialization in Minimal Language Model Populations."* Every commit and experiment is now aligned with specific Research Questions (RQs) and thesis chapters.
