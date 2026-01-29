# Phase 2: Temporal Emergence Dynamics Study

**Design Document**
**Created**: 2026-01-28
**Status**: Planning - Ready for Implementation

---

## Overview

Phase 2 extends the initial multi-agent system experiments to study **temporal dynamics of emergence**: how specialization develops over time, when it stabilizes, and whether learned specialization persists without continued reinforcement.

This addresses a critical validity question: **Is the observed specialization genuine emergent behavior, or is it mechanically induced by routing + few-shot prompting?**

---

## Research Questions

**RQ1 (Stability)**: Does learned specialization persist when context accumulation stops (frozen agents)?

**RQ2 (Development)**: How does specialization evolve over extended training (50+ questions)? Do we observe:
- Gradual drift (linear increase)?
- Phase transitions (sudden jumps)?
- Plateaus (stable equilibria)?
- Destabilization (collapse)?

**RQ3 (Baseline)**: What is the performance contribution of the multi-agent framework vs. single-model baseline?

**RQ4 (Generalization)**: Does specialization transfer to new, unseen tasks from the same distribution?

---

## Experimental Conditions

### Condition 0: Single Model Baseline (No Framework)

**Purpose**: Establish absolute performance ceiling without any multi-agent system.

**Setup**:
- Single Qwen2.5-Coder model (1.5B or 7B - match Phase 1 choice)
- No agent abstraction, no context accumulation
- Direct model inference on tasks

**Execution**:
- Run on 300 SWEBench Lite questions sequentially
- No learning between tasks (stateless)
- Measure Pass@1, tokens/solution

**Deliverables**:
- Baseline performance metrics
- Cost analysis (tokens, time)

---

### Condition 1: Current Framework Baseline (Continuous Learning)

**Purpose**: Measure emergence over extended training without branching.

**Setup**:
- Your existing multi-agent system (N agents, configuration from Phase 1)
- Continuous context accumulation for 300 questions
- No checkpointing/branching (simplified execution)

**Execution**:
- Run 300 SWEBench Lite questions with full routing + learning
- Compute metrics (S, D, F) every 10 questions
- Log all task assignments and performance

**Deliverables**:
- Complete emergence timeline: S(t), D(t), F(t) over 300 questions
- Identification of when/if specialization emerges
- Performance trajectory vs. Condition 0

**Critical Question**: Does S > 0.2 with p < 0.05 by question 300? If not, framework may not induce specialization.

---

### Condition 2: Adaptive Checkpointing (Frozen vs Continued)

**Purpose**: Test stability and development hypotheses through branching.

**Setup**:
- Same as Condition 1, but with adaptive checkpoint triggers
- At each checkpoint, spawn two branches:
  - **Frozen**: Contexts locked, evaluate on held-out set
  - **Continued**: Contexts keep learning, proceed to next checkpoint

**Execution**: See "Checkpoint System" section below

**Deliverables**:
- Frozen branch generalization performance at each checkpoint
- Continued branch emergence trajectory
- Comparative analysis: frozen vs continued performance over time

---

## Infrastructure Components

### 1. Dataset Manager

**Requirements**:
- Load full SWEBench Lite dataset (300 questions)
- Extract category metadata from dataset
- Maintain stratified sampling pools per category
- Track used vs unused questions

**Category Distribution** (from SWEBench Lite):
- Load actual distribution when implementing
- Expected categories: bug, feature, test, refactor, docs, other
- Ensure evaluation sets match training distribution

**Implementation Notes**:
```python
class DatasetManager:
    def __init__(self, dataset_path):
        self.full_dataset = load_swebench_lite()
        self.category_pools = self._stratify_by_category()
        self.used_questions = set()

    def sample_evaluation_set(self, n=50):
        """Sample n questions maintaining category distribution"""
        pass

    def get_next_training_batch(self, n):
        """Get next n questions for training"""
        pass

    def mark_used(self, question_ids):
        """Track used questions to prevent reuse"""
        pass
```

---

### 2. Metrics Monitor

**Requirements**:
- Compute S, D, F after every measurement window (e.g., 10 questions)
- Maintain history for trend analysis
- Implement all metrics from metrics-document.md

**Implementation Notes**:
```python
class MetricsMonitor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def update(self, task_log, agent_contexts):
        """Compute current metrics"""
        S = self.specialization_index(task_log)
        D = self.context_divergence(agent_contexts)
        F = self.functional_differentiation(task_log)

        metrics = {
            'S': S,
            'D': D,
            'F_pvalue': F['p_value'],
            'F_significant': F['significant'],
            'pass_at_1': self._compute_pass_at_1(task_log),
            'questions_processed': len(task_log)
        }

        self.history.append(metrics)
        return metrics

    def specialization_index(self, task_log):
        """Implementation from metrics-document.md"""
        pass

    def context_divergence(self, agent_contexts):
        """Implementation from metrics-document.md"""
        pass

    def functional_differentiation(self, task_log):
        """Implementation from metrics-document.md"""
        pass
```

---

### 3. Checkpoint Trigger System

**Requirements**:
- Monitor metrics after each measurement window
- Detect positive triggers (emergence events)
- Detect negative triggers (destabilization events)
- Enforce minimum spacing (20 questions) and fallback intervals (50 questions)

**Trigger Logic**:

**Positive Triggers** (Emergence):
```python
POSITIVE_TRIGGERS = {
    'phase_transition': lambda curr, prev: curr['S'] - prev['S'] > 0.15,
    'sustained_divergence': lambda history: _check_monotonic_increase(history, 'D', 5),
    'functional_significance': lambda curr, prev: curr['F_pvalue'] < 0.05 and prev['F_pvalue'] >= 0.05,
}
```

**Negative Triggers** (Destabilization):
```python
NEGATIVE_TRIGGERS = {
    'collapse': lambda curr, prev: curr['S'] - prev['S'] < -0.15,
    'convergence': lambda curr, prev: curr['D'] - prev['D'] < -0.10,
    'performance_drop': lambda curr, prev: curr['pass_at_1'] < prev['pass_at_1'] - 0.15,
}
```

**Fallback**:
```python
FALLBACK_TRIGGERS = {
    'fixed_interval': lambda questions_since: questions_since >= 50,
    'minimum_spacing': lambda questions_since: questions_since >= 20,
}
```

**Implementation Notes**:
```python
class CheckpointTrigger:
    def __init__(self, metrics_monitor):
        self.metrics_monitor = metrics_monitor
        self.questions_since_last = 0
        self.checkpoint_count = 0

    def should_checkpoint(self) -> dict:
        """
        Returns: {
            'checkpoint': bool,
            'reason': str,
            'type': 'positive' | 'negative' | 'fallback',
            'metrics': dict
        }
        """
        history = self.metrics_monitor.history

        # Minimum spacing enforcement
        if self.questions_since_last < 20:
            return {'checkpoint': False}

        # Check positive triggers
        # Check negative triggers
        # Check fallback

        pass
```

---

### 4. Branch Manager

**Requirements**:
- Capture complete system state at checkpoint
- Create frozen and continued branches
- Execute frozen branch evaluation (batch mode)
- Continue training on continued branch

**State Capture**:
```python
class SystemState:
    """Complete serializable system state"""
    def __init__(self, agents, task_log, metrics):
        self.agent_contexts = {aid: agent.context for aid, agent in agents.items()}
        self.routing_state = {aid: agent.affinity_scores for aid, agent in agents.items()}
        self.task_history = task_log
        self.metrics = metrics
        self.timestamp = datetime.now()

    def save(self, checkpoint_dir):
        """Serialize to JSON"""
        pass

    @classmethod
    def load(cls, checkpoint_dir):
        """Deserialize from JSON"""
        pass
```

**Branch Execution**:
```python
class BranchManager:
    def __init__(self, dataset_manager, system):
        self.dataset_manager = dataset_manager
        self.system = system
        self.checkpoints = []

    def create_checkpoint(self, trigger_info):
        """Capture state and spawn branches"""
        checkpoint_id = len(self.checkpoints)
        state = SystemState(self.system.agents,
                           self.system.task_log,
                           trigger_info['metrics'])

        # Save checkpoint
        checkpoint_dir = f"checkpoints/checkpoint_{checkpoint_id}"
        state.save(checkpoint_dir)

        # Spawn frozen branch
        frozen_branch = self._create_frozen_branch(state, checkpoint_id)

        # Continued branch is current system (keeps going)

        self.checkpoints.append({
            'id': checkpoint_id,
            'state': state,
            'frozen_branch': frozen_branch,
            'trigger': trigger_info
        })

        return checkpoint_id

    def _create_frozen_branch(self, state, checkpoint_id):
        """Create and execute frozen evaluation branch"""
        # Sample evaluation set (stratified, 50 questions)
        eval_questions = self.dataset_manager.sample_evaluation_set(n=50)

        # Restore system state with frozen contexts
        frozen_system = self._restore_system(state, contexts_locked=True)

        # Run evaluation
        results = frozen_system.evaluate(eval_questions)

        # Save results
        frozen_dir = f"checkpoints/checkpoint_{checkpoint_id}/frozen"
        self._save_results(results, frozen_dir)

        return {
            'eval_questions': eval_questions,
            'results': results,
            'metrics': self._compute_frozen_metrics(results)
        }
```

---

### 5. Experiment Runner

**Requirements**:
- Orchestrate all three conditions
- Handle logging and result aggregation
- Provide progress tracking
- Handle crashes/resumption

**Implementation Notes**:
```python
class Phase2Experiment:
    def __init__(self, config):
        self.config = config
        self.dataset_manager = DatasetManager(config.dataset_path)
        self.metrics_monitor = MetricsMonitor(window_size=10)
        self.checkpoint_trigger = CheckpointTrigger(self.metrics_monitor)
        self.branch_manager = BranchManager(self.dataset_manager, None)

    def run_condition_0_single_baseline(self):
        """Run single model baseline"""
        print("=== Condition 0: Single Model Baseline ===")
        # Load model directly, no agent framework
        # Run 300 questions
        # Log results
        pass

    def run_condition_1_continuous_learning(self):
        """Run current framework, no branching"""
        print("=== Condition 1: Continuous Learning Baseline ===")
        # Initialize multi-agent system
        # Run 300 questions with metric tracking
        # No checkpointing/branching
        pass

    def run_condition_2_adaptive_checkpointing(self):
        """Run adaptive checkpointing with frozen/continued branches"""
        print("=== Condition 2: Adaptive Checkpointing ===")

        system = self._initialize_system()
        self.branch_manager.system = system

        questions_processed = 0
        training_pool = self.dataset_manager.get_training_pool(300)

        for question in training_pool:
            # Process question
            result = system.process(question)
            questions_processed += 1

            # Update metrics every window_size
            if questions_processed % self.metrics_monitor.window_size == 0:
                metrics = self.metrics_monitor.update(
                    system.task_log,
                    system.get_agent_contexts()
                )

                # Check for checkpoint trigger
                trigger_decision = self.checkpoint_trigger.should_checkpoint()

                if trigger_decision['checkpoint']:
                    print(f"Checkpoint triggered at Q{questions_processed}: {trigger_decision['reason']}")
                    checkpoint_id = self.branch_manager.create_checkpoint(trigger_decision)
                    self.checkpoint_trigger.questions_since_last = 0
                else:
                    self.checkpoint_trigger.questions_since_last += self.metrics_monitor.window_size

        # Final checkpoint at end
        final_metrics = self.metrics_monitor.update(system.task_log, system.get_agent_contexts())
        self.branch_manager.create_checkpoint({
            'checkpoint': True,
            'reason': 'Final checkpoint',
            'type': 'fallback',
            'metrics': final_metrics
        })

    def run_all(self):
        """Execute all conditions sequentially"""
        self.run_condition_0_single_baseline()
        self.run_condition_1_continuous_learning()
        self.run_condition_2_adaptive_checkpointing()
```

---

## Implementation Phases

### Phase 2.1: Infrastructure Setup (Week 1)

**Tasks**:
1. Implement DatasetManager
   - Load full SWEBench Lite
   - Extract category distributions
   - Implement stratified sampling
2. Implement MetricsMonitor
   - Port all metrics from metrics-document.md
   - Add embedding model for context divergence
3. Implement CheckpointTrigger
   - All trigger conditions
   - Minimum spacing logic
4. Implement BranchManager
   - State serialization
   - Branch creation
5. Implement Phase2Experiment runner

**Deliverable**: Complete infrastructure, unit tested

---

### Phase 2.2: Condition 0 - Single Baseline (Week 1-2)

**Tasks**:
1. Run single model on 300 SWEBench Lite questions
2. Measure Pass@1, tokens, time
3. Generate baseline report

**Deliverable**: `results/phase2/condition0_single_baseline.json`

---

### Phase 2.3: Condition 1 - Continuous Learning (Week 2)

**Tasks**:
1. Run multi-agent system on 300 questions
2. Track S(t), D(t), F(t) every 10 questions
3. Generate emergence timeline plots
4. Analyze: Does specialization emerge?

**Deliverable**: `results/phase2/condition1_continuous_learning.json` + plots

---

### Phase 2.4: Condition 2 - Adaptive Checkpointing (Week 3-4)

**Tasks**:
1. Run full experiment with checkpointing
2. Execute all frozen branches
3. Collect comparative data
4. Analyze frozen vs continued dynamics

**Deliverable**: `results/phase2/condition2_checkpointing/` with all checkpoint data

---

### Phase 2.5: Analysis & Visualization (Week 4)

**Tasks**:
1. Statistical analysis of all hypotheses
2. Generate comparison plots:
   - Condition 0 vs 1 vs 2 performance
   - Emergence timelines
   - Frozen vs continued trajectories
   - Checkpoint trigger patterns
3. Write results chapter section

**Deliverable**: Phase 2 analysis report + figures

---

## Data Requirements

### Storage Estimates

**Per Condition**:
- Condition 0: ~100MB (logs, results)
- Condition 1: ~500MB (metrics history, task logs)
- Condition 2: ~2-3GB (multiple checkpoints + frozen branches)

**Total**: ~4GB for full Phase 2

### Checkpoints Expected

With 300 questions and adaptive triggers:
- Minimum: 6 checkpoints (50-question fallback intervals)
- Realistic: 8-12 checkpoints (mix of triggers)
- Maximum: 15 checkpoints (if many phase transitions)

---

## Metrics and Analysis

### Primary Outcomes

**H1 (Stability)**: Frozen branch Pass@1 remains within 5% of checkpoint performance
- **Measure**: Compare frozen_branch.pass_at_1 vs continued_branch.pass_at_1_at_checkpoint
- **Statistical test**: Paired t-test across checkpoints

**H2 (Development)**: S(t) shows phase transition pattern (not gradual drift)
- **Measure**: Detect sudden jumps using phase transition algorithm
- **Classification**: Gradual drift vs phase transition vs oscillation

**H3 (Framework Benefit)**: Condition 1 > Condition 0
- **Measure**: Pass@1 improvement, token efficiency
- **Statistical test**: Independent t-test

**H4 (Generalization)**: Frozen branch performance on unseen tasks
- **Measure**: Frozen branch Pass@1 on stratified evaluation sets
- **Analysis**: Per-category breakdown

### Secondary Analysis

- **Checkpoint trigger patterns**: Which triggers predict successful emergence?
- **Negative signals**: When does destabilization occur?
- **Population dynamics**: Which agents specialize in which categories?
- **Efficiency**: Tokens per solution across conditions

---

## Expected Outputs

### Results Directory Structure

```
results/phase2/
├── condition0_single_baseline/
│   ├── results.json
│   ├── metrics.json
│   └── logs/
├── condition1_continuous_learning/
│   ├── results.json
│   ├── emergence_timeline.json
│   ├── metrics_history.json
│   └── plots/
│       ├── S_over_time.png
│       ├── D_over_time.png
│       └── F_over_time.png
├── condition2_checkpointing/
│   ├── summary.json
│   ├── checkpoints/
│   │   ├── checkpoint_0/
│   │   │   ├── state.json
│   │   │   ├── frozen/
│   │   │   │   ├── eval_set.json
│   │   │   │   ├── results.json
│   │   │   │   └── metrics.json
│   │   │   └── trigger_info.json
│   │   ├── checkpoint_1/
│   │   └── ...
│   └── plots/
│       ├── frozen_vs_continued.png
│       ├── checkpoint_triggers.png
│       └── generalization_by_category.png
└── analysis/
    ├── hypothesis_tests.json
    ├── comparative_analysis.md
    └── figures/
```

### Key Visualizations

1. **Emergence Timeline**: S(t), D(t), F(t) over 300 questions with checkpoint markers
2. **Frozen vs Continued**: Performance trajectories for both branches
3. **Checkpoint Analysis**: Trigger types and their outcomes
4. **Category Breakdown**: Specialization patterns by task category
5. **Condition Comparison**: Pass@1 across all three conditions

---

## Stretch Goals (Optional Extensions)

### Extension A: Temporal Stability Testing (Approach C)

Instead of batch evaluation in frozen branches, spread evaluation over time:

**Implementation**:
- At checkpoint, run 10 eval questions immediately
- Every 25 questions in continued branch, run 10 more frozen eval questions
- Continue until 50 frozen eval questions exhausted

**Analysis**:
- Plot frozen branch performance over time
- Test: Does performance degrade without reinforcement?
- Measure: "Half-life" of specialization

**Deliverable**: Temporal stability curves

---

### Extension B: Pre-trained Specialization Experiment

Address original concern: Can we differentiate pre-training effects from emergent dynamics?

**Setup**:
- Run Condition 1 to 50 questions → agents differentiate
- Fine-tune individual agents on their specialized categories (LoRA adapters)
- Run Phase 2 with pre-trained specialists

**Analysis**:
- Compare emergence with identical vs differentiated starting points
- Test: Does context accumulation matter if models start biased?

**Deliverable**: Pre-training comparison report

---

### Extension C: Ablation Studies

**Variations to test**:
1. **No routing**: Random agent assignment, still accumulate context
2. **No context**: Routing based on affinity but no few-shot examples
3. **Fixed context**: Context from first 20 questions, then frozen

**Analysis**: Isolate contribution of routing vs context accumulation

---

## Dependencies

### Python Packages
```
datasets>=2.14.0  # For loading SWEBench Lite
sentence-transformers>=2.2.0  # For context embeddings
scipy>=1.9.0  # For statistical tests
matplotlib>=3.5.0  # For plots
pandas>=1.5.0  # For data analysis
```

### Data
- SWEBench Lite (300 questions) from HuggingFace datasets
- Existing model checkpoints from Phase 1

### Compute
- Same as Phase 1: Single GPU (8-16GB VRAM)
- Estimated time: 1-2 weeks wall-clock for all conditions

---

## Success Criteria

**Minimum Viable Phase 2**:
- ✅ All 3 conditions executed
- ✅ Complete metrics collected
- ✅ At least 6 checkpoints in Condition 2
- ✅ Statistical tests for H1-H4

**Ideal Phase 2**:
- ✅ Minimum viable +
- ✅ 10+ checkpoints with mix of positive/negative triggers
- ✅ Clear phase transition detected in S(t)
- ✅ Frozen branches demonstrate generalization
- ✅ All visualizations generated

**Publication-Ready Phase 2**:
- ✅ Ideal +
- ✅ At least one stretch goal completed
- ✅ Negative results analyzed (if emergence doesn't occur)
- ✅ Comprehensive discussion of limitations

---

## Timeline Estimate

**Conservative** (4 weeks):
- Week 1: Infrastructure + Condition 0
- Week 2: Condition 1
- Week 3-4: Condition 2 + Analysis

**Optimistic** (2-3 weeks):
- Week 1: Infrastructure + Conditions 0-1
- Week 2: Condition 2
- Week 3: Analysis + Stretch goal

**Risk Buffer**: Add 1-2 weeks for debugging, unexpected findings, or re-runs

---

## Next Steps

1. **Review this plan** with advisor/committee for feedback
2. **Finalize configuration**: Lock agent count, temperature, prompt from Phase 1
3. **Download SWEBench Lite**: Verify dataset access and category distribution
4. **Implement infrastructure** (Phase 2.1)
5. **Execute experiments** sequentially (Phases 2.2-2.4)
6. **Analyze and write** (Phase 2.5)

---

**Document Version**: 1.0
**Status**: ✅ Design Complete - Ready for Implementation Review
