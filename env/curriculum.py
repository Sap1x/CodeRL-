"""
CodeRL Adaptive Curriculum — Performance-based task selection.

Selects tasks based on agent performance history, implementing:
    - Easy task retirement (once agent scores >= threshold)
    - Progressive hard task introduction
    - Mistake-targeted task selection (targets agent's weak spots)
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Optional

from env.state import AgentMemory, Difficulty, Task
from env.task_loader import TaskLoader


# ──────────────────────────────────────────────
# Strategy Enum
# ──────────────────────────────────────────────

class CurriculumStrategy(str, Enum):
    """Task selection strategy."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    PERFORMANCE_ADAPTIVE = "performance_adaptive"


# ──────────────────────────────────────────────
# Configuration Defaults
# ──────────────────────────────────────────────

DEFAULT_EASY_RETIRE_THRESHOLD = 0.8
DEFAULT_HARD_INTRO_THRESHOLD = 0.6
DEFAULT_MISTAKE_TARGET_WEIGHT = 0.3


# ──────────────────────────────────────────────
# Adaptive Curriculum
# ──────────────────────────────────────────────

class AdaptiveCurriculum:
    """
    Selects tasks based on agent performance, implementing the
    self-improving curriculum described in CodeRL++.

    Behaviors:
        - Easy tasks are retired once the agent reliably scores >= threshold
        - Hard adversarial patterns are introduced progressively
        - Mistake-targeted tasks focus on the agent's known weak spots
        - Task complexity scales with agent capability
    """

    def __init__(
        self,
        task_loader: TaskLoader,
        strategy: CurriculumStrategy = CurriculumStrategy.PERFORMANCE_ADAPTIVE,
        easy_retire_threshold: float = DEFAULT_EASY_RETIRE_THRESHOLD,
        hard_intro_threshold: float = DEFAULT_HARD_INTRO_THRESHOLD,
        mistake_target_weight: float = DEFAULT_MISTAKE_TARGET_WEIGHT,
    ):
        self._task_loader = task_loader
        self._strategy = strategy
        self._easy_retire_threshold = easy_retire_threshold
        self._hard_intro_threshold = hard_intro_threshold
        self._mistake_target_weight = mistake_target_weight

        # Track per-task scores
        self._task_scores: dict[str, list[float]] = {}
        self._sequential_index = 0

    # ======================================================================
    # Public API
    # ======================================================================

    def select_next_task(
        self,
        memory: Optional[AgentMemory] = None,
    ) -> Task:
        """
        Select the next task based on the configured strategy.

        Args:
            memory: Agent's cross-episode memory for adaptive selection

        Returns:
            The selected Task
        """
        if self._strategy == CurriculumStrategy.SEQUENTIAL:
            return self._select_sequential()
        elif self._strategy == CurriculumStrategy.RANDOM:
            return self._select_random()
        else:
            return self._select_adaptive(memory)

    def record_task_score(self, task_id: str, score: float) -> None:
        """Record a score for a completed task."""
        if task_id not in self._task_scores:
            self._task_scores[task_id] = []
        self._task_scores[task_id].append(score)

    def get_curriculum_state(self) -> dict[str, Any]:
        """Return the current curriculum state for diagnostics."""
        return {
            "strategy": self._strategy.value,
            "task_scores": {
                tid: {
                    "attempts": len(scores),
                    "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                    "best_score": round(max(scores), 4) if scores else 0.0,
                    "retired": self._is_task_retired(tid),
                }
                for tid, scores in self._task_scores.items()
            },
            "active_tasks": [t.id for t in self._get_active_tasks(None)],
            "retired_tasks": [
                tid for tid in self._task_scores if self._is_task_retired(tid)
            ],
        }

    # ======================================================================
    # Selection Strategies
    # ======================================================================

    def _select_sequential(self) -> Task:
        """Select tasks in order."""
        all_ids = self._task_loader.get_all_task_ids()
        task_id = all_ids[self._sequential_index % len(all_ids)]
        self._sequential_index += 1
        return self._task_loader.get_task(task_id)

    def _select_random(self) -> Task:
        """Select a random task."""
        all_ids = self._task_loader.get_all_task_ids()
        task_id = random.choice(all_ids)
        return self._task_loader.get_task(task_id)

    def _select_adaptive(self, memory: Optional[AgentMemory] = None) -> Task:
        """
        Performance-adaptive task selection.

        Priority order:
        1. Mistake-targeted tasks (if agent has known weaknesses)
        2. Tasks at the agent's current difficulty frontier
        3. Random from non-retired tasks
        """
        active_tasks = self._get_active_tasks(memory)

        if not active_tasks:
            # All tasks retired — reset and try again
            self._task_scores.clear()
            active_tasks = self._task_loader.get_all_tasks()

        # ── Mistake-targeted selection ──
        if memory and random.random() < self._mistake_target_weight:
            targeted = self._select_mistake_targeted(active_tasks, memory)
            if targeted:
                return targeted

        # ── Difficulty-frontier selection ──
        return self._select_frontier(active_tasks)

    # ======================================================================
    # Internal Helpers
    # ======================================================================

    def _get_active_tasks(self, memory: Optional[AgentMemory]) -> list[Task]:
        """Get tasks that haven't been retired."""
        all_tasks = self._task_loader.get_all_tasks()

        # Filter retired easy tasks
        active = [t for t in all_tasks if not self._is_task_retired(t.id)]

        # Filter hard tasks if agent isn't ready
        if memory and memory.improvement_trajectory:
            avg_score = sum(
                e.score for e in memory.improvement_trajectory[-10:]
            ) / min(len(memory.improvement_trajectory), 10)

            if avg_score < self._hard_intro_threshold:
                active = [t for t in active if t.difficulty != Difficulty.HARD]

        return active if active else all_tasks

    def _is_task_retired(self, task_id: str) -> bool:
        """Check if a task has been retired (agent consistently scores well)."""
        scores = self._task_scores.get(task_id, [])
        if len(scores) < 3:
            return False

        # Task considers difficulty — only retire easy tasks
        try:
            task = self._task_loader.get_task(task_id)
            if task.difficulty != Difficulty.EASY:
                return False
        except KeyError:
            return False

        # Check if last 3 scores are all above threshold
        recent = scores[-3:]
        return all(s >= self._easy_retire_threshold for s in recent)

    def _select_mistake_targeted(
        self,
        active_tasks: list[Task],
        memory: AgentMemory,
    ) -> Optional[Task]:
        """Select a task that targets the agent's known weaknesses."""
        if not memory.missed_vulnerabilities:
            return None

        # Find the most commonly missed vulnerability types
        missed_types: dict[str, int] = {}
        for v in memory.missed_vulnerabilities[-20:]:
            missed_types[v.issue_type] = missed_types.get(v.issue_type, 0) + 1

        if not missed_types:
            return None

        # Find tasks containing these vulnerability types
        top_missed = sorted(missed_types.items(), key=lambda x: -x[1])[:3]
        top_missed_set = {t[0] for t in top_missed}

        scored_tasks: list[tuple[int, Task]] = []
        for task in active_tasks:
            overlap = sum(
                1 for gt in task.ground_truth if gt.issue in top_missed_set
            )
            if overlap > 0:
                scored_tasks.append((overlap, task))

        if scored_tasks:
            # Weight by overlap count
            scored_tasks.sort(key=lambda x: -x[0])
            return scored_tasks[0][1]

        return None

    def _select_frontier(self, active_tasks: list[Task]) -> Task:
        """Select a task at the agent's current difficulty frontier."""
        # Group by difficulty
        by_difficulty: dict[Difficulty, list[Task]] = {
            Difficulty.EASY: [],
            Difficulty.MEDIUM: [],
            Difficulty.HARD: [],
        }
        for t in active_tasks:
            by_difficulty[t.difficulty].append(t)

        # Prefer medium, then hard, then easy
        priority = [Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EASY]
        for diff in priority:
            candidates = by_difficulty[diff]
            if candidates:
                # Prefer tasks with fewer attempts
                candidates.sort(
                    key=lambda t: len(self._task_scores.get(t.id, []))
                )
                return candidates[0]

        # Fallback
        return random.choice(active_tasks)
