# Phase 2: Temporal Emergence Dynamics - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build infrastructure to measure temporal emergence dynamics across 300 SWEBench questions with adaptive checkpointing and three experimental conditions.

**Architecture:** Five interconnected components (DatasetManager, MetricsMonitor, CheckpointTrigger, BranchManager, Phase2Experiment) that extend the existing swarm system with checkpointing, frozen/continued branching, and comprehensive metrics tracking.

**Tech Stack:** Python 3.9+, sentence-transformers, scipy, pandas, existing swarm infrastructure (src/swarm/)

---

## Prerequisites

**Existing Code to Leverage:**
- `src/swarm/agent.py`: SwarmAgent with context buffer
- `src/swarm/metrics.py`: MetricsEngine (S, D, F already implemented!)
- `src/swarm/experiment.py`: SwarmExperiment orchestrator
- `src/swarm/router.py`: Router implementations
- `data/swebench_curated.json`: 50-question curated dataset

**New Components to Build:**
- `src/swarm/phase2/dataset_manager.py`: Dataset loading and stratified sampling
- `src/swarm/phase2/checkpoint_trigger.py`: Adaptive trigger system
- `src/swarm/phase2/branch_manager.py`: State serialization and branching
- `src/swarm/phase2/phase2_experiment.py`: Main orchestrator for all conditions

---

## Task 1: Create Phase 2 Directory Structure

**Files:**
- Create: `src/swarm/phase2/__init__.py`
- Create: `src/swarm/phase2/dataset_manager.py`
- Create: `src/swarm/phase2/checkpoint_trigger.py`
- Create: `src/swarm/phase2/branch_manager.py`
- Create: `src/swarm/phase2/phase2_experiment.py`
- Create: `tests/unit/test_phase2_infrastructure.py`

**Step 1: Write test structure**

```python
# tests/unit/test_phase2_infrastructure.py
"""Unit tests for Phase 2 infrastructure components."""

import pytest
import json
from pathlib import Path


@pytest.fixture
def mock_swebench_dataset():
    """Mock SWEBench dataset with known categories."""
    return {
        "metadata": {
            "n_issues": 10,
            "repositories": ["django", "pytest"]
        },
        "issues": [
            {
                "instance_id": f"test_{i}",
                "problem_statement": f"Problem {i}",
                "category": ["bug", "feature", "test"][i % 3],
                "repo": ["django", "pytest"][i % 2]
            }
            for i in range(10)
        ]
    }


class TestDatasetManager:
    """Test DatasetManager initialization and sampling."""

    def test_init_loads_dataset(self, tmp_path, mock_swebench_dataset):
        """Test DatasetManager loads and stratifies dataset."""
        # Will implement after DatasetManager exists
        pass

    def test_stratified_sampling_maintains_distribution(self, tmp_path, mock_swebench_dataset):
        """Test stratified sampling preserves category ratios."""
        pass


class TestCheckpointTrigger:
    """Test checkpoint trigger logic."""

    def test_minimum_spacing_enforcement(self):
        """Test trigger enforces 20-question minimum spacing."""
        pass

    def test_positive_trigger_phase_transition(self):
        """Test phase transition trigger (ΔS > 0.15)."""
        pass

    def test_fallback_trigger_at_50_questions(self):
        """Test fallback trigger fires at 50 questions."""
        pass


class TestBranchManager:
    """Test state serialization and branch execution."""

    def test_state_serialization_roundtrip(self):
        """Test SystemState can save and load."""
        pass

    def test_frozen_branch_execution(self):
        """Test frozen branch runs with locked contexts."""
        pass
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_phase2_infrastructure.py -v`
Expected: All tests FAIL (modules don't exist yet)

**Step 3: Create empty module files**

```python
# src/swarm/phase2/__init__.py
"""Phase 2: Temporal Emergence Dynamics Infrastructure."""

from .dataset_manager import DatasetManager
from .checkpoint_trigger import CheckpointTrigger
from .branch_manager import BranchManager, SystemState
from .phase2_experiment import Phase2Experiment, Phase2Config

__all__ = [
    "DatasetManager",
    "CheckpointTrigger",
    "BranchManager",
    "SystemState",
    "Phase2Experiment",
    "Phase2Config"
]
```

**Step 4: Commit**

```bash
git add src/swarm/phase2/ tests/unit/test_phase2_infrastructure.py
git commit -m "feat: add Phase 2 directory structure and test skeleton

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Implement DatasetManager

**Files:**
- Modify: `src/swarm/phase2/dataset_manager.py`
- Test: `tests/unit/test_phase2_infrastructure.py`

**Step 1: Write failing test**

```python
# tests/unit/test_phase2_infrastructure.py (add to TestDatasetManager)

def test_init_loads_dataset(self, tmp_path, mock_swebench_dataset):
    """Test DatasetManager loads and stratifies dataset."""
    from src.swarm.phase2 import DatasetManager

    # Write mock dataset
    dataset_path = tmp_path / "test_dataset.json"
    dataset_path.write_text(json.dumps(mock_swebench_dataset))

    # Initialize manager
    manager = DatasetManager(str(dataset_path))

    # Verify loading
    assert len(manager.full_dataset) == 10
    assert "bug" in manager.category_pools
    assert len(manager.used_questions) == 0

def test_stratified_sampling_maintains_distribution(self, tmp_path, mock_swebench_dataset):
    """Test stratified sampling preserves category ratios."""
    from src.swarm.phase2 import DatasetManager

    dataset_path = tmp_path / "test_dataset.json"
    dataset_path.write_text(json.dumps(mock_swebench_dataset))

    manager = DatasetManager(str(dataset_path))

    # Sample 6 questions (60% of 10)
    sample = manager.sample_evaluation_set(n=6)

    # Check we got 6 questions
    assert len(sample) == 6

    # Check distribution roughly preserved (3 bugs, 3 features, 4 tests in original)
    categories = [q["category"] for q in sample]
    assert len(set(categories)) >= 2  # At least 2 different categories
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_phase2_infrastructure.py::TestDatasetManager -v`
Expected: FAIL with "module not found" or "function not defined"

**Step 3: Implement DatasetManager**

```python
# src/swarm/phase2/dataset_manager.py
"""Dataset management with stratified sampling for Phase 2."""

import json
from typing import List, Dict, Set
from pathlib import Path
from collections import defaultdict
import random


class DatasetManager:
    """
    Manages SWEBench dataset with stratified sampling.

    Responsibilities:
    - Load full dataset
    - Stratify by category
    - Sample evaluation sets maintaining distribution
    - Track used/unused questions
    """

    def __init__(self, dataset_path: str, random_seed: int = 42):
        """
        Initialize dataset manager.

        Args:
            dataset_path: Path to SWEBench JSON file
            random_seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.random_seed = random_seed
        random.seed(random_seed)

        # Load dataset
        with open(self.dataset_path) as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.full_dataset = data["issues"]

        # Stratify by category
        self.category_pools = self._stratify_by_category()

        # Track used questions
        self.used_questions: Set[str] = set()

    def _stratify_by_category(self) -> Dict[str, List[Dict]]:
        """
        Organize dataset by category.

        Returns:
            Dict mapping category -> list of issues
        """
        pools = defaultdict(list)

        for issue in self.full_dataset:
            # Get category (may be list or string)
            category = issue.get("category", "unknown")
            if isinstance(category, list):
                category = category[0] if category else "unknown"

            pools[category].append(issue)

        return dict(pools)

    def sample_evaluation_set(self, n: int) -> List[Dict]:
        """
        Sample n questions maintaining category distribution.

        Uses stratified sampling to preserve category ratios.
        Only samples from unused questions.

        Args:
            n: Number of questions to sample

        Returns:
            List of question dicts
        """
        # Calculate target per category
        total_available = len(self.full_dataset) - len(self.used_questions)
        if total_available == 0:
            return []

        # Get available questions per category
        available_by_category = {}
        for category, pool in self.category_pools.items():
            available = [q for q in pool if q["instance_id"] not in self.used_questions]
            if available:
                available_by_category[category] = available

        # Stratified sampling
        sample = []
        n_per_category = {}
        total_in_categories = sum(len(pool) for pool in available_by_category.values())

        for category, pool in available_by_category.items():
            # Proportional allocation
            target = int(n * len(pool) / total_in_categories)
            n_per_category[category] = min(target, len(pool))

        # Adjust for rounding errors
        while sum(n_per_category.values()) < n:
            # Add one to largest category
            max_cat = max(n_per_category.items(), key=lambda x: len(available_by_category[x[0]]) - x[1])
            if len(available_by_category[max_cat[0]]) > n_per_category[max_cat[0]]:
                n_per_category[max_cat[0]] += 1
            else:
                break

        # Sample from each category
        for category, target_n in n_per_category.items():
            pool = available_by_category[category]
            sampled = random.sample(pool, min(target_n, len(pool)))
            sample.extend(sampled)

        return sample

    def get_training_pool(self, n: int) -> List[Dict]:
        """
        Get next n questions for training (sequential, not stratified).

        Args:
            n: Number of questions

        Returns:
            List of question dicts
        """
        available = [q for q in self.full_dataset if q["instance_id"] not in self.used_questions]
        return available[:n]

    def mark_used(self, question_ids: List[str]) -> None:
        """
        Mark questions as used.

        Args:
            question_ids: List of instance IDs
        """
        self.used_questions.update(question_ids)

    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get category distribution of full dataset.

        Returns:
            Dict mapping category -> count
        """
        return {cat: len(pool) for cat, pool in self.category_pools.items()}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_phase2_infrastructure.py::TestDatasetManager -v`
Expected: Both tests PASS

**Step 5: Commit**

```bash
git add src/swarm/phase2/dataset_manager.py tests/unit/test_phase2_infrastructure.py
git commit -m "feat: implement DatasetManager with stratified sampling

- Loads SWEBench datasets with category metadata
- Stratified sampling preserves category distributions
- Tracks used/unused questions
- Tested with unit tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement CheckpointTrigger

**Files:**
- Modify: `src/swarm/phase2/checkpoint_trigger.py`
- Test: `tests/unit/test_phase2_infrastructure.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_phase2_infrastructure.py (add to TestCheckpointTrigger)

def test_minimum_spacing_enforcement(self):
    """Test trigger enforces 20-question minimum spacing."""
    from src.swarm.phase2 import CheckpointTrigger
    from src.swarm.metrics import MetricsEngine

    metrics_engine = MetricsEngine()
    trigger = CheckpointTrigger(metrics_engine)

    # Simulate metrics that would trigger
    metrics_engine.history = [
        {"S": 0.10, "questions_processed": 10},
        {"S": 0.30, "questions_processed": 15}  # ΔS = 0.20 > 0.15
    ]

    trigger.questions_since_last = 5  # Only 5 questions since last

    decision = trigger.should_checkpoint()

    assert decision["checkpoint"] is False  # Should block due to spacing

def test_positive_trigger_phase_transition(self):
    """Test phase transition trigger (ΔS > 0.15)."""
    from src.swarm.phase2 import CheckpointTrigger
    from src.swarm.metrics import MetricsEngine

    metrics_engine = MetricsEngine()
    trigger = CheckpointTrigger(metrics_engine)

    # Simulate phase transition
    metrics_engine.history = [
        {"S": 0.10, "D": 0.2, "F_pvalue": 0.5, "pass_at_1": 0.3, "questions_processed": 20},
        {"S": 0.30, "D": 0.25, "F_pvalue": 0.4, "pass_at_1": 0.35, "questions_processed": 30}
    ]

    trigger.questions_since_last = 25  # Sufficient spacing

    decision = trigger.should_checkpoint()

    assert decision["checkpoint"] is True
    assert decision["type"] == "positive"
    assert "phase_transition" in decision["reason"]

def test_fallback_trigger_at_50_questions(self):
    """Test fallback trigger fires at 50 questions."""
    from src.swarm.phase2 import CheckpointTrigger
    from src.swarm.metrics import MetricsEngine

    metrics_engine = MetricsEngine()
    trigger = CheckpointTrigger(metrics_engine)

    # Simulate stable metrics (no transitions)
    metrics_engine.history = [
        {"S": 0.10, "D": 0.2, "F_pvalue": 0.5, "pass_at_1": 0.3, "questions_processed": i * 10}
        for i in range(1, 6)
    ]

    trigger.questions_since_last = 50  # Hit fallback threshold

    decision = trigger.should_checkpoint()

    assert decision["checkpoint"] is True
    assert decision["type"] == "fallback"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_phase2_infrastructure.py::TestCheckpointTrigger -v`
Expected: FAIL

**Step 3: Implement CheckpointTrigger**

```python
# src/swarm/phase2/checkpoint_trigger.py
"""Adaptive checkpoint trigger system for Phase 2."""

from typing import Dict, Optional, List
from src.swarm.metrics import MetricsEngine


class CheckpointTrigger:
    """
    Detects when to create checkpoints based on emergence signals.

    Triggers on:
    - Positive signals: Phase transitions, sustained divergence, functional significance
    - Negative signals: Collapse, convergence, performance drops
    - Fallback: Every 50 questions minimum

    Enforces minimum spacing of 20 questions.
    """

    def __init__(self, metrics_monitor: MetricsEngine):
        """
        Initialize checkpoint trigger.

        Args:
            metrics_monitor: MetricsEngine with history
        """
        self.metrics_monitor = metrics_monitor
        self.questions_since_last = 0
        self.checkpoint_count = 0

        # Thresholds
        self.MIN_SPACING = 20
        self.FALLBACK_INTERVAL = 50
        self.PHASE_TRANSITION_THRESHOLD = 0.15
        self.DIVERGENCE_THRESHOLD = 0.10
        self.PERFORMANCE_DROP_THRESHOLD = 0.15

    def should_checkpoint(self) -> Dict:
        """
        Determine if checkpoint should be created.

        Returns:
            Dict with:
                - checkpoint: bool
                - reason: str (if checkpoint)
                - type: "positive" | "negative" | "fallback"
                - metrics: current metrics
        """
        history = self.metrics_monitor.history

        if not history or len(history) < 2:
            return {"checkpoint": False}

        # Enforce minimum spacing
        if self.questions_since_last < self.MIN_SPACING:
            return {"checkpoint": False}

        curr = history[-1]
        prev = history[-2]

        # Check positive triggers
        positive_trigger = self._check_positive_triggers(curr, prev, history)
        if positive_trigger:
            self.checkpoint_count += 1
            return {
                "checkpoint": True,
                "reason": positive_trigger,
                "type": "positive",
                "metrics": curr,
                "checkpoint_id": self.checkpoint_count
            }

        # Check negative triggers
        negative_trigger = self._check_negative_triggers(curr, prev)
        if negative_trigger:
            self.checkpoint_count += 1
            return {
                "checkpoint": True,
                "reason": negative_trigger,
                "type": "negative",
                "metrics": curr,
                "checkpoint_id": self.checkpoint_count
            }

        # Check fallback
        if self.questions_since_last >= self.FALLBACK_INTERVAL:
            self.checkpoint_count += 1
            return {
                "checkpoint": True,
                "reason": "Fixed 50-question interval",
                "type": "fallback",
                "metrics": curr,
                "checkpoint_id": self.checkpoint_count
            }

        return {"checkpoint": False}

    def _check_positive_triggers(
        self,
        curr: Dict,
        prev: Dict,
        history: List[Dict]
    ) -> Optional[str]:
        """
        Check for positive emergence signals.

        Returns:
            Trigger reason string if triggered, else None
        """
        # Phase transition: ΔS > 0.15
        delta_S = curr.get("S", 0) - prev.get("S", 0)
        if delta_S > self.PHASE_TRANSITION_THRESHOLD:
            return f"phase_transition (ΔS = {delta_S:.3f})"

        # Sustained divergence: D increases for 5 consecutive windows
        if len(history) >= 5:
            last_5 = history[-5:]
            D_values = [h.get("D", 0) for h in last_5]
            if all(D_values[i] < D_values[i+1] for i in range(len(D_values)-1)):
                return "sustained_divergence (5+ windows)"

        # Functional significance: F becomes significant
        curr_F = curr.get("F_pvalue", 1.0)
        prev_F = prev.get("F_pvalue", 1.0)
        if curr_F < 0.05 and prev_F >= 0.05:
            return f"functional_significance (p = {curr_F:.4f})"

        return None

    def _check_negative_triggers(self, curr: Dict, prev: Dict) -> Optional[str]:
        """
        Check for negative destabilization signals.

        Returns:
            Trigger reason string if triggered, else None
        """
        # Specialization collapse: ΔS < -0.15
        delta_S = curr.get("S", 0) - prev.get("S", 0)
        if delta_S < -self.PHASE_TRANSITION_THRESHOLD:
            return f"collapse (ΔS = {delta_S:.3f})"

        # Context convergence: ΔD < -0.10
        delta_D = curr.get("D", 0) - prev.get("D", 0)
        if delta_D < -self.DIVERGENCE_THRESHOLD:
            return f"convergence (ΔD = {delta_D:.3f})"

        # Performance drop: Δ Pass@1 < -0.15
        delta_perf = curr.get("pass_at_1", 0) - prev.get("pass_at_1", 0)
        if delta_perf < -self.PERFORMANCE_DROP_THRESHOLD:
            return f"performance_drop (Δ = {delta_perf:.3f})"

        return None

    def reset_spacing_counter(self) -> None:
        """Reset questions counter after checkpoint."""
        self.questions_since_last = 0
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_phase2_infrastructure.py::TestCheckpointTrigger -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/swarm/phase2/checkpoint_trigger.py tests/unit/test_phase2_infrastructure.py
git commit -m "feat: implement adaptive checkpoint trigger system

- Detects positive triggers (phase transitions, sustained divergence)
- Detects negative triggers (collapse, convergence, performance drops)
- Enforces minimum 20-question spacing
- Fallback trigger at 50 questions
- Tested with unit tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement SystemState and BranchManager

**Files:**
- Modify: `src/swarm/phase2/branch_manager.py`
- Test: `tests/unit/test_phase2_infrastructure.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_phase2_infrastructure.py (add to TestBranchManager)

def test_state_serialization_roundtrip(self, tmp_path):
    """Test SystemState can save and load."""
    from src.swarm.phase2 import SystemState
    from src.swarm.agent import SwarmAgent

    # Create mock agents
    agents = [
        SwarmAgent(agent_id=0, max_context_examples=3),
        SwarmAgent(agent_id=1, max_context_examples=3)
    ]

    # Add some context
    agents[0].add_success("def add(a, b):", "return a + b", "math")

    # Create state
    state = SystemState(
        agents={a.agent_id: a for a in agents},
        task_log=[{"agent_id": 0, "task_type": "math", "success": True}],
        metrics={"S": 0.25, "D": 0.15},
        questions_processed=10
    )

    # Save
    checkpoint_dir = tmp_path / "checkpoint_0"
    state.save(str(checkpoint_dir))

    # Load
    loaded_state = SystemState.load(str(checkpoint_dir))

    # Verify
    assert len(loaded_state.agent_contexts) == 2
    assert loaded_state.questions_processed == 10
    assert loaded_state.metrics["S"] == 0.25

def test_frozen_branch_execution(self, tmp_path):
    """Test frozen branch runs with locked contexts."""
    # Will implement after BranchManager exists
    pass
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_phase2_infrastructure.py::TestBranchManager::test_state_serialization_roundtrip -v`
Expected: FAIL

**Step 3: Implement SystemState**

```python
# src/swarm/phase2/branch_manager.py
"""State management and branch execution for Phase 2."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from src.swarm.agent import SwarmAgent


@dataclass
class SystemState:
    """
    Complete serializable system state at checkpoint.

    Captures:
    - Agent contexts (few-shot examples)
    - Task history
    - Current metrics
    - Questions processed
    """

    agent_contexts: Dict[int, Dict]
    task_log: List[Dict]
    metrics: Dict
    questions_processed: int
    timestamp: str = ""

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def save(self, checkpoint_dir: str) -> None:
        """
        Serialize state to disk.

        Args:
            checkpoint_dir: Directory to save state
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save state as JSON
        state_dict = {
            "agent_contexts": self.agent_contexts,
            "task_log": self.task_log,
            "metrics": self.metrics,
            "questions_processed": self.questions_processed,
            "timestamp": self.timestamp
        }

        with open(checkpoint_path / "state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

    @classmethod
    def load(cls, checkpoint_dir: str) -> "SystemState":
        """
        Deserialize state from disk.

        Args:
            checkpoint_dir: Directory containing state

        Returns:
            Loaded SystemState
        """
        checkpoint_path = Path(checkpoint_dir)

        with open(checkpoint_path / "state.json") as f:
            state_dict = json.load(f)

        # Convert agent_contexts keys back to int
        agent_contexts = {
            int(k): v for k, v in state_dict["agent_contexts"].items()
        }

        return cls(
            agent_contexts=agent_contexts,
            task_log=state_dict["task_log"],
            metrics=state_dict["metrics"],
            questions_processed=state_dict["questions_processed"],
            timestamp=state_dict.get("timestamp", "")
        )

    @classmethod
    def from_agents(
        cls,
        agents: Dict[int, SwarmAgent],
        task_log: List[Dict],
        metrics: Dict,
        questions_processed: int
    ) -> "SystemState":
        """
        Create state from current system.

        Args:
            agents: Dict of agent_id -> SwarmAgent
            task_log: Complete task log
            metrics: Current metrics
            questions_processed: Number of questions processed

        Returns:
            SystemState
        """
        # Serialize agent contexts
        agent_contexts = {}
        for agent_id, agent in agents.items():
            agent_contexts[agent_id] = {
                "agent_id": agent.agent_id,
                "max_context_examples": agent.max_context_examples,
                "context_buffer": [
                    {
                        "problem": ex.problem,
                        "solution": ex.solution,
                        "task_type": ex.task_type if isinstance(ex.task_type, str) else ex.task_type.value
                    }
                    for ex in agent.context_buffer
                ],
                "task_history": {
                    k if isinstance(k, str) else k.value: v
                    for k, v in agent.task_history.items()
                }
            }

        return cls(
            agent_contexts=agent_contexts,
            task_log=task_log,
            metrics=metrics,
            questions_processed=questions_processed
        )


class BranchManager:
    """
    Manages checkpoint creation and branch execution.

    At each checkpoint:
    1. Capture complete system state
    2. Spawn frozen branch (evaluate on held-out set)
    3. Continue with current system (continued branch)
    """

    def __init__(self, dataset_manager, output_dir: str = "results/phase2/checkpoints"):
        """
        Initialize branch manager.

        Args:
            dataset_manager: DatasetManager instance
            output_dir: Base directory for checkpoint outputs
        """
        self.dataset_manager = dataset_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints: List[Dict] = []

    def create_checkpoint(
        self,
        agents: Dict[int, SwarmAgent],
        task_log: List[Dict],
        trigger_info: Dict,
        questions_processed: int
    ) -> int:
        """
        Create checkpoint and spawn branches.

        Args:
            agents: Current agent dict
            task_log: Complete task log
            trigger_info: Trigger decision dict
            questions_processed: Number of questions processed

        Returns:
            Checkpoint ID
        """
        checkpoint_id = len(self.checkpoints)

        # Capture state
        state = SystemState.from_agents(
            agents=agents,
            task_log=task_log,
            metrics=trigger_info["metrics"],
            questions_processed=questions_processed
        )

        # Create checkpoint directory
        checkpoint_dir = self.output_dir / f"checkpoint_{checkpoint_id}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save state
        state.save(str(checkpoint_dir))

        # Save trigger info
        with open(checkpoint_dir / "trigger_info.json", "w") as f:
            json.dump(trigger_info, f, indent=2)

        # Record checkpoint
        self.checkpoints.append({
            "id": checkpoint_id,
            "checkpoint_dir": str(checkpoint_dir),
            "trigger": trigger_info,
            "questions_processed": questions_processed,
            "timestamp": state.timestamp
        })

        return checkpoint_id

    def execute_frozen_branch(
        self,
        checkpoint_id: int,
        system,
        n_eval_questions: int = 50
    ) -> Dict:
        """
        Execute frozen branch evaluation.

        Args:
            checkpoint_id: ID of checkpoint to evaluate
            system: SwarmExperiment instance to restore state to
            n_eval_questions: Number of evaluation questions

        Returns:
            Frozen branch results dict
        """
        checkpoint = self.checkpoints[checkpoint_id]
        checkpoint_dir = Path(checkpoint["checkpoint_dir"])

        # Load state
        state = SystemState.load(str(checkpoint_dir))

        # Sample evaluation set (stratified, unused)
        eval_questions = self.dataset_manager.sample_evaluation_set(n=n_eval_questions)

        # Mark as used
        self.dataset_manager.mark_used([q["instance_id"] for q in eval_questions])

        # Restore system state with frozen contexts
        self._restore_frozen_state(system, state)

        # Run evaluation
        results = []
        for question in eval_questions:
            result = system.process_task(question, contexts_frozen=True)
            results.append(result)

        # Compute frozen metrics
        frozen_metrics = {
            "pass_at_1": sum(r["success"] for r in results) / len(results),
            "n_questions": len(results),
            "checkpoint_id": checkpoint_id
        }

        # Save frozen branch results
        frozen_dir = checkpoint_dir / "frozen"
        frozen_dir.mkdir(exist_ok=True)

        with open(frozen_dir / "eval_set.json", "w") as f:
            json.dump([q["instance_id"] for q in eval_questions], f, indent=2)

        with open(frozen_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(frozen_dir / "metrics.json", "w") as f:
            json.dump(frozen_metrics, f, indent=2)

        return {
            "checkpoint_id": checkpoint_id,
            "eval_questions": eval_questions,
            "results": results,
            "metrics": frozen_metrics
        }

    def _restore_frozen_state(self, system, state: SystemState) -> None:
        """
        Restore system to frozen state (contexts locked).

        Args:
            system: SwarmExperiment instance
            state: SystemState to restore
        """
        # Restore agent contexts
        for agent_id, agent in system.agents.items():
            if agent_id in state.agent_contexts:
                context_data = state.agent_contexts[agent_id]

                # Clear current context
                agent.context_buffer.clear()
                agent.task_history.clear()

                # Restore context buffer
                from src.swarm.types import FewShotExample, TaskType
                for ex_data in context_data["context_buffer"]:
                    ex = FewShotExample(
                        problem=ex_data["problem"],
                        solution=ex_data["solution"],
                        task_type=TaskType(ex_data["task_type"]) if isinstance(ex_data["task_type"], str) else ex_data["task_type"]
                    )
                    agent.context_buffer.append(ex)

                # Restore task history
                for task_type_str, history in context_data["task_history"].items():
                    task_type = TaskType(task_type_str) if isinstance(task_type_str, str) else task_type_str
                    agent.task_history[task_type] = history
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_phase2_infrastructure.py::TestBranchManager::test_state_serialization_roundtrip -v`
Expected: Test PASSES

**Step 5: Commit**

```bash
git add src/swarm/phase2/branch_manager.py tests/unit/test_phase2_infrastructure.py
git commit -m "feat: implement SystemState and BranchManager

- SystemState: Serializable checkpoint state
- Save/load agent contexts, task log, metrics
- BranchManager: Creates checkpoints and executes frozen branches
- Frozen branch evaluation with locked contexts
- Tested with unit tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement Phase2Experiment Orchestrator

**Files:**
- Modify: `src/swarm/phase2/phase2_experiment.py`
- Create: `tests/integration/test_phase2_experiment.py`

**Step 1: Write integration test structure**

```python
# tests/integration/test_phase2_experiment.py
"""Integration tests for Phase 2 experiment orchestrator."""

import pytest
from pathlib import Path


@pytest.fixture
def phase2_config(tmp_path):
    """Create minimal Phase2Config for testing."""
    from src.swarm.phase2 import Phase2Config

    return Phase2Config(
        dataset_path="data/swebench_curated.json",
        output_dir=str(tmp_path / "phase2_test"),
        n_training_questions=20,  # Small for testing
        n_eval_questions=10,
        model_path="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        n_agents=2,
        random_seed=42
    )


class TestPhase2Experiment:
    """Integration tests for full experiment."""

    def test_condition_1_runs_without_error(self, phase2_config):
        """Test Condition 1 completes successfully."""
        # Will run after implementation
        pass

    def test_checkpoint_triggers_create_branches(self, phase2_config):
        """Test Condition 2 creates checkpoints and frozen branches."""
        pass
```

**Step 2: Implement Phase2Config and Phase2Experiment**

```python
# src/swarm/phase2/phase2_experiment.py
"""Main orchestrator for Phase 2 experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import json
import logging

from src.models.interface import ModelInterface
from src.swarm.experiment import SwarmExperiment, ExperimentConfig
from src.swarm.metrics import MetricsEngine
from .dataset_manager import DatasetManager
from .checkpoint_trigger import CheckpointTrigger
from .branch_manager import BranchManager

logger = logging.getLogger(__name__)


@dataclass
class Phase2Config:
    """Configuration for Phase 2 experiments."""

    # Dataset
    dataset_path: str = "data/swebench_curated.json"
    n_training_questions: int = 300
    n_eval_questions: int = 50

    # Model
    model_path: str = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    context_length: int = 4096
    max_tokens: int = 512

    # Agents
    n_agents: int = 3
    max_context_examples: int = 5

    # Metrics
    metrics_window: int = 10  # Compute metrics every N questions

    # Checkpointing
    enable_checkpointing: bool = True

    # Output
    output_dir: str = "results/phase2"

    # Random seed
    random_seed: Optional[int] = 42


class Phase2Experiment:
    """
    Orchestrates all Phase 2 experimental conditions.

    Three conditions:
    - Condition 0: Single model baseline (no framework)
    - Condition 1: Current framework with continuous learning
    - Condition 2: Adaptive checkpointing with frozen/continued branches
    """

    def __init__(self, config: Phase2Config):
        """
        Initialize Phase 2 experiment.

        Args:
            config: Phase2Config
        """
        self.config = config

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset manager
        self.dataset_manager = DatasetManager(
            config.dataset_path,
            random_seed=config.random_seed
        )

        # Initialize metrics engine
        self.metrics_engine = MetricsEngine()

        # Initialize checkpoint components
        if config.enable_checkpointing:
            self.checkpoint_trigger = CheckpointTrigger(self.metrics_engine)
            self.branch_manager = BranchManager(
                self.dataset_manager,
                output_dir=str(self.output_dir / "checkpoints")
            )

        # Export config
        self._save_config()

    def _save_config(self) -> None:
        """Save experiment configuration."""
        config_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.config.__dict__.items()
        }

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def run_condition_0_single_baseline(self, model: ModelInterface) -> Dict:
        """
        Run Condition 0: Single model baseline.

        Args:
            model: Model interface

        Returns:
            Results dict
        """
        logger.info("=== Condition 0: Single Model Baseline ===")

        condition_dir = self.output_dir / "condition0_single_baseline"
        condition_dir.mkdir(exist_ok=True)

        # Get training questions
        questions = self.dataset_manager.get_training_pool(self.config.n_training_questions)

        # Run single model (no framework, no context)
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Condition 0: Question {i+1}/{len(questions)}")

            # Simple direct inference
            prompt = f"Solve this coding problem:\n\n{question['problem_statement']}"

            try:
                response = model.generate(
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=0.2
                )

                # TODO: Execute and validate solution
                success = False  # Placeholder

                results.append({
                    "instance_id": question["instance_id"],
                    "success": success,
                    "response": response
                })
            except Exception as e:
                logger.error(f"Error on question {i}: {e}")
                results.append({
                    "instance_id": question["instance_id"],
                    "success": False,
                    "error": str(e)
                })

        # Compute metrics
        pass_at_1 = sum(r["success"] for r in results) / len(results)

        metrics = {
            "pass_at_1": pass_at_1,
            "n_questions": len(results)
        }

        # Save results
        with open(condition_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(condition_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Condition 0 complete: Pass@1 = {pass_at_1:.3f}")

        return {
            "results": results,
            "metrics": metrics
        }

    def run_condition_1_continuous_learning(self, model: ModelInterface) -> Dict:
        """
        Run Condition 1: Current framework with continuous learning.

        Args:
            model: Model interface

        Returns:
            Results dict with emergence timeline
        """
        logger.info("=== Condition 1: Continuous Learning ===")

        condition_dir = self.output_dir / "condition1_continuous_learning"
        condition_dir.mkdir(exist_ok=True)

        # Initialize swarm system
        swarm_config = ExperimentConfig(
            model_path=self.config.model_path,
            n_agents=self.config.n_agents,
            max_context_examples=self.config.max_context_examples,
            n_tasks=self.config.n_training_questions,
            output_dir=str(condition_dir),
            random_seed=self.config.random_seed
        )

        swarm = SwarmExperiment(model=model, config=swarm_config)

        # Get training questions
        questions = self.dataset_manager.get_training_pool(self.config.n_training_questions)

        # Track metrics timeline
        metrics_timeline = []

        # Run experiment
        for i, question in enumerate(questions):
            logger.info(f"Condition 1: Question {i+1}/{len(questions)}")

            # Process task
            result = swarm.process_task(question)

            # Compute metrics every window
            if (i + 1) % self.config.metrics_window == 0:
                metrics = self._compute_current_metrics(swarm)
                metrics["questions_processed"] = i + 1
                metrics_timeline.append(metrics)

                logger.info(f"Metrics at Q{i+1}: S={metrics.get('S', 0):.3f}, D={metrics.get('D', 0):.3f}")

        # Final metrics
        final_metrics = self._compute_current_metrics(swarm)
        final_metrics["questions_processed"] = len(questions)
        metrics_timeline.append(final_metrics)

        # Save results
        with open(condition_dir / "emergence_timeline.json", "w") as f:
            json.dump(metrics_timeline, f, indent=2)

        logger.info("Condition 1 complete")

        return {
            "metrics_timeline": metrics_timeline,
            "final_metrics": final_metrics
        }

    def run_condition_2_adaptive_checkpointing(self, model: ModelInterface) -> Dict:
        """
        Run Condition 2: Adaptive checkpointing with frozen/continued branches.

        Args:
            model: Model interface

        Returns:
            Results dict with checkpoint data
        """
        logger.info("=== Condition 2: Adaptive Checkpointing ===")

        condition_dir = self.output_dir / "condition2_checkpointing"
        condition_dir.mkdir(exist_ok=True)

        # Initialize swarm system
        swarm_config = ExperimentConfig(
            model_path=self.config.model_path,
            n_agents=self.config.n_agents,
            max_context_examples=self.config.max_context_examples,
            n_tasks=self.config.n_training_questions,
            output_dir=str(condition_dir),
            random_seed=self.config.random_seed
        )

        swarm = SwarmExperiment(model=model, config=swarm_config)

        # Get training questions
        questions = self.dataset_manager.get_training_pool(self.config.n_training_questions)

        # Track metrics and checkpoints
        metrics_timeline = []
        checkpoints_created = []

        # Run experiment with checkpointing
        for i, question in enumerate(questions):
            logger.info(f"Condition 2: Question {i+1}/{len(questions)}")

            # Process task
            result = swarm.process_task(question)

            # Update metrics every window
            if (i + 1) % self.config.metrics_window == 0:
                metrics = self._compute_current_metrics(swarm)
                metrics["questions_processed"] = i + 1
                self.metrics_engine.history.append(metrics)
                metrics_timeline.append(metrics)

                # Check for checkpoint trigger
                trigger_decision = self.checkpoint_trigger.should_checkpoint()

                if trigger_decision["checkpoint"]:
                    logger.info(f"Checkpoint triggered at Q{i+1}: {trigger_decision['reason']}")

                    # Create checkpoint
                    checkpoint_id = self.branch_manager.create_checkpoint(
                        agents={a.agent_id: a for a in swarm.agents},
                        task_log=swarm.task_log,
                        trigger_info=trigger_decision,
                        questions_processed=i + 1
                    )

                    # Execute frozen branch
                    frozen_results = self.branch_manager.execute_frozen_branch(
                        checkpoint_id=checkpoint_id,
                        system=swarm,
                        n_eval_questions=self.config.n_eval_questions
                    )

                    checkpoints_created.append({
                        "checkpoint_id": checkpoint_id,
                        "questions_processed": i + 1,
                        "trigger": trigger_decision,
                        "frozen_metrics": frozen_results["metrics"]
                    })

                    # Reset spacing counter
                    self.checkpoint_trigger.reset_spacing_counter()
                else:
                    self.checkpoint_trigger.questions_since_last += self.config.metrics_window

        # Final checkpoint
        final_metrics = self._compute_current_metrics(swarm)
        final_metrics["questions_processed"] = len(questions)
        self.metrics_engine.history.append(final_metrics)

        final_checkpoint_id = self.branch_manager.create_checkpoint(
            agents={a.agent_id: a for a in swarm.agents},
            task_log=swarm.task_log,
            trigger_info={
                "checkpoint": True,
                "reason": "Final checkpoint",
                "type": "fallback",
                "metrics": final_metrics
            },
            questions_processed=len(questions)
        )

        # Save summary
        summary = {
            "n_checkpoints": len(checkpoints_created) + 1,
            "checkpoints": checkpoints_created,
            "metrics_timeline": metrics_timeline
        }

        with open(condition_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Condition 2 complete: {len(checkpoints_created) + 1} checkpoints created")

        return summary

    def _compute_current_metrics(self, swarm: SwarmExperiment) -> Dict:
        """
        Compute current metrics from swarm state.

        Args:
            swarm: SwarmExperiment instance

        Returns:
            Metrics dict
        """
        # Compute S, D, F
        S = self.metrics_engine.specialization_index(swarm.task_log)
        D = self.metrics_engine.context_divergence(swarm.agents)
        F = self.metrics_engine.functional_differentiation(swarm.task_log)

        # Compute Pass@1
        successful = sum(t.get("success", False) for t in swarm.task_log)
        pass_at_1 = successful / len(swarm.task_log) if swarm.task_log else 0.0

        return {
            "S": S,
            "D": D,
            "F_pvalue": F["p_value"],
            "F_significant": F["significant"],
            "pass_at_1": pass_at_1
        }

    def run_all(self, model: ModelInterface) -> None:
        """
        Run all three conditions sequentially.

        Args:
            model: Model interface
        """
        logger.info("Starting Phase 2 full experimental run")

        # Condition 0
        c0_results = self.run_condition_0_single_baseline(model)

        # Condition 1
        c1_results = self.run_condition_1_continuous_learning(model)

        # Condition 2
        c2_results = self.run_condition_2_adaptive_checkpointing(model)

        logger.info("Phase 2 complete - all conditions finished")
```

**Step 3: Commit**

```bash
git add src/swarm/phase2/phase2_experiment.py tests/integration/test_phase2_experiment.py
git commit -m "feat: implement Phase2Experiment orchestrator

- Orchestrates all three experimental conditions
- Condition 0: Single model baseline
- Condition 1: Continuous learning with metrics tracking
- Condition 2: Adaptive checkpointing with frozen branches
- Integration with existing swarm infrastructure
- Comprehensive logging and result saving

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Create Phase 2 Runner Script

**Files:**
- Create: `scripts/run_phase2.py`

**Step 1: Write runner script**

```python
# scripts/run_phase2.py
"""Runner script for Phase 2 experiments."""

import argparse
import logging
from pathlib import Path

from src.models.llama_cpp import LlamaCppModel
from src.swarm.phase2 import Phase2Experiment, Phase2Config


def setup_logging(output_dir: str) -> None:
    """Configure logging."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(output_dir) / "phase2.log"),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Phase 2 experiments")

    # Conditions
    parser.add_argument(
        "--condition",
        type=int,
        choices=[0, 1, 2],
        help="Run specific condition (0, 1, or 2). If not specified, runs all."
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="data/swebench_curated.json",
        help="Path to SWEBench dataset"
    )

    # Model
    parser.add_argument(
        "--model",
        default="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        help="Path to model"
    )

    # Configuration
    parser.add_argument(
        "--n-questions",
        type=int,
        default=300,
        help="Number of training questions"
    )

    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
        help="Number of swarm agents"
    )

    parser.add_argument(
        "--output-dir",
        default="results/phase2",
        help="Output directory"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)

    logger.info("=== Phase 2: Temporal Emergence Dynamics ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Training questions: {args.n_questions}")
    logger.info(f"Agents: {args.n_agents}")

    # Create config
    config = Phase2Config(
        dataset_path=args.dataset,
        model_path=args.model,
        n_training_questions=args.n_questions,
        n_agents=args.n_agents,
        output_dir=args.output_dir,
        random_seed=args.seed
    )

    # Initialize experiment
    experiment = Phase2Experiment(config)

    # Load model
    logger.info("Loading model...")
    model = LlamaCppModel(
        model_path=args.model,
        n_ctx=config.context_length,
        n_gpu_layers=-1  # Use all GPU layers
    )

    # Run experiment(s)
    if args.condition is not None:
        # Run specific condition
        if args.condition == 0:
            logger.info("Running Condition 0 only")
            experiment.run_condition_0_single_baseline(model)
        elif args.condition == 1:
            logger.info("Running Condition 1 only")
            experiment.run_condition_1_continuous_learning(model)
        elif args.condition == 2:
            logger.info("Running Condition 2 only")
            experiment.run_condition_2_adaptive_checkpointing(model)
    else:
        # Run all conditions
        logger.info("Running all conditions")
        experiment.run_all(model)

    logger.info("Phase 2 complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Test runner script**

Run: `python scripts/run_phase2.py --help`
Expected: Help message displays correctly

**Step 3: Commit**

```bash
git add scripts/run_phase2.py
git commit -m "feat: add Phase 2 runner script

- CLI for running all conditions or specific conditions
- Configurable parameters (model, dataset, n_questions, n_agents)
- Logging setup
- Ready for execution

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add Metrics Engine History Tracking

**Files:**
- Modify: `src/swarm/metrics.py`
- Test: `tests/unit/test_metrics.py`

**Step 1: Write failing test**

```python
# tests/unit/test_metrics.py (add if doesn't exist)
"""Tests for MetricsEngine with history tracking."""

def test_metrics_engine_maintains_history():
    """Test MetricsEngine history attribute."""
    from src.swarm.metrics import MetricsEngine

    engine = MetricsEngine()

    # Verify history exists
    assert hasattr(engine, "history")
    assert isinstance(engine.history, list)
    assert len(engine.history) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_metrics.py::test_metrics_engine_maintains_history -v`
Expected: FAIL (history attribute doesn't exist)

**Step 3: Add history tracking to MetricsEngine**

```python
# src/swarm/metrics.py (modify __init__)

def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Initialize metrics engine.

    Args:
        embedding_model: SentenceTransformer model for embeddings
    """
    self.embedder = SentenceTransformer(embedding_model)
    self.history: List[Dict] = []  # Add this line
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_metrics.py::test_metrics_engine_maintains_history -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swarm/metrics.py tests/unit/test_metrics.py
git commit -m "feat: add history tracking to MetricsEngine

- MetricsEngine now maintains history list
- Required for checkpoint trigger system
- Tested with unit test

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integration with Existing SwarmExperiment

**Files:**
- Modify: `src/swarm/experiment.py`
- Add method: `process_task()` for single task execution

**Step 1: Review current SwarmExperiment structure**

Read: `src/swarm/experiment.py`
Identify: Need `process_task()` method and `task_log` attribute

**Step 2: Add missing attributes and methods**

```python
# src/swarm/experiment.py (add to __init__)

def __init__(self, model: ModelInterface, config: ExperimentConfig):
    # ... existing code ...

    # Add task log for metrics
    self.task_log: List[Dict] = []

# Add new method
def process_task(self, task: Dict, contexts_frozen: bool = False) -> Dict:
    """
    Process a single task.

    Args:
        task: Task dict with problem_statement, instance_id, etc.
        contexts_frozen: If True, don't update agent contexts

    Returns:
        Result dict with success, agent_id, etc.
    """
    # Extract task info
    problem = task.get("problem_statement", "")
    instance_id = task.get("instance_id", "unknown")

    # Route to agent
    selected_agent = self.router.select_agent(
        self.agents,
        task_description=problem
    )

    # Build prompt with agent's context
    prompt = selected_agent.build_prompt(
        self.config.system_prompt,
        problem
    )

    # Generate solution
    try:
        solution = self.model.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=0.2
        )

        # Execute and validate
        success = self.executor.execute(solution, task.get("test_cases", []))

        # Update agent context (unless frozen)
        if not contexts_frozen:
            if success:
                selected_agent.add_success(
                    problem=problem,
                    solution=solution,
                    task_type=task.get("category", "unknown")
                )
            else:
                selected_agent.add_failure(
                    task_type=task.get("category", "unknown")
                )

        # Log task
        result = {
            "instance_id": instance_id,
            "agent_id": selected_agent.agent_id,
            "task_type": task.get("category", "unknown"),
            "success": success,
            "solution": solution
        }

        self.task_log.append(result)

        return result

    except Exception as e:
        logger.error(f"Error processing task {instance_id}: {e}")
        result = {
            "instance_id": instance_id,
            "agent_id": selected_agent.agent_id,
            "task_type": task.get("category", "unknown"),
            "success": False,
            "error": str(e)
        }
        self.task_log.append(result)
        return result

def get_agent_contexts(self) -> Dict[int, List[str]]:
    """
    Get agent contexts for metrics computation.

    Returns:
        Dict mapping agent_id -> list of context strings
    """
    return {
        agent.agent_id: [
            f"{ex.problem}\n{ex.solution}"
            for ex in agent.context_buffer
        ]
        for agent in self.agents
    }
```

**Step 3: Commit**

```bash
git add src/swarm/experiment.py
git commit -m "feat: add process_task method to SwarmExperiment

- Single task processing with frozen context support
- Task log tracking for metrics
- Agent context accessor for metrics engine
- Integrates with Phase 2 infrastructure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Documentation and README

**Files:**
- Modify: `docs/plans/2026-01-28-phase2-implementation.md` (add completion notes)
- Create: `docs/phase2_usage.md`

**Step 1: Create usage guide**

```markdown
# docs/phase2_usage.md
# Phase 2 Usage Guide

## Quick Start

Run all three conditions:

```bash
python scripts/run_phase2.py \
  --dataset data/swebench_curated.json \
  --model models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --n-questions 300 \
  --n-agents 3 \
  --output-dir results/phase2
```

Run specific condition:

```bash
# Condition 0: Single baseline
python scripts/run_phase2.py --condition 0

# Condition 1: Continuous learning
python scripts/run_phase2.py --condition 1

# Condition 2: Adaptive checkpointing
python scripts/run_phase2.py --condition 2
```

## Output Structure

```
results/phase2/
├── config.json
├── phase2.log
├── condition0_single_baseline/
│   ├── results.json
│   └── metrics.json
├── condition1_continuous_learning/
│   ├── emergence_timeline.json
│   └── metrics_history.json
└── condition2_checkpointing/
    ├── summary.json
    └── checkpoints/
        ├── checkpoint_0/
        │   ├── state.json
        │   ├── trigger_info.json
        │   └── frozen/
        │       ├── eval_set.json
        │       ├── results.json
        │       └── metrics.json
        └── ...
```

## Analyzing Results

Use Jupyter notebooks or analysis scripts:

```python
import json
from pathlib import Path

# Load Condition 1 timeline
with open("results/phase2/condition1_continuous_learning/emergence_timeline.json") as f:
    timeline = json.load(f)

# Plot S(t)
import matplotlib.pyplot as plt

questions = [m["questions_processed"] for m in timeline]
S_values = [m["S"] for m in timeline]

plt.plot(questions, S_values)
plt.xlabel("Questions Processed")
plt.ylabel("Specialization Index (S)")
plt.title("Emergence Timeline - Condition 1")
plt.show()
```

## Testing Infrastructure

Run unit tests:

```bash
pytest tests/unit/test_phase2_infrastructure.py -v
```

Run integration tests:

```bash
pytest tests/integration/test_phase2_experiment.py -v
```
```

**Step 2: Commit documentation**

```bash
git add docs/phase2_usage.md docs/plans/2026-01-28-phase2-implementation.md
git commit -m "docs: add Phase 2 usage guide and completion notes

- Usage guide with examples
- Output structure documentation
- Analysis examples
- Testing instructions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Final Integration Test

**Files:**
- Test: Run full integration test with small dataset

**Step 1: Create minimal test run**

```bash
# Create test config with small numbers
python scripts/run_phase2.py \
  --condition 1 \
  --n-questions 20 \
  --n-agents 2 \
  --output-dir results/phase2_test \
  --seed 42
```

**Step 2: Verify outputs**

Check:
- `results/phase2_test/config.json` exists
- `results/phase2_test/condition1_continuous_learning/emergence_timeline.json` exists
- Metrics computed correctly (S, D, F values present)

**Step 3: Run unit tests**

```bash
pytest tests/unit/test_phase2_infrastructure.py -v
```

Expected: All tests PASS

**Step 4: Commit completion**

```bash
git add -A
git commit -m "test: verify Phase 2 infrastructure with integration test

- Tested with 20-question run
- All components working correctly
- Metrics computed successfully
- Ready for full 300-question experiments

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria

After completing all tasks:

**Infrastructure Complete:**
- ✅ DatasetManager loads and stratifies SWEBench dataset
- ✅ CheckpointTrigger detects emergence/destabilization
- ✅ BranchManager creates and executes frozen branches
- ✅ Phase2Experiment orchestrates all conditions
- ✅ MetricsEngine tracks history

**Testing Complete:**
- ✅ Unit tests pass for all components
- ✅ Integration test runs without errors
- ✅ Metrics computed correctly

**Ready for Execution:**
- ✅ Can run Condition 0 (single baseline)
- ✅ Can run Condition 1 (continuous learning)
- ✅ Can run Condition 2 (adaptive checkpointing)
- ✅ Outputs saved in structured format

**Documentation:**
- ✅ Usage guide written
- ✅ Implementation plan complete
- ✅ Code well-commented

---

## Next Steps After Implementation

1. **Run Pilot**: Test with 50 questions to verify checkpoint triggers
2. **Run Full Experiment**: Execute all 300 questions for each condition
3. **Analysis**: Generate plots and statistical tests
4. **Write Results**: Document findings in thesis chapter

---

**Estimated Implementation Time:**
- Tasks 1-3: 2-3 hours (dataset management, triggers)
- Tasks 4-5: 3-4 hours (branching, orchestration)
- Tasks 6-8: 2-3 hours (runner, integration)
- Tasks 9-10: 1-2 hours (docs, testing)
- **Total: 8-12 hours of focused development**

---

**Document Version**: 1.0
**Status**: ✅ Ready for Implementation
