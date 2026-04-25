"""
CodeRL Agent Memory Manager — Cross-episode memory for self-improving agents.

Tracks patterns of mistakes, false positives, and reasoning failures
across episodes, enabling the improvement reward signal and adaptive
curriculum selection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from env.grader import GradeResult
from env.state import (
    AgentMemory,
    EpisodeScore,
    FalsePositiveRecord,
    ImprovementReward,
    Phase,
    ReasoningFailure,
    ReviewComment,
    Severity,
    ToolStats,
    VulnerabilityRecord,
)

logger = logging.getLogger("coderl.memory")


# ──────────────────────────────────────────────
# Default improvement reward coefficients
# ──────────────────────────────────────────────

DEFAULT_ALPHA = 0.3   # recall gain weight
DEFAULT_BETA = 0.4    # repeated mistake penalty weight
DEFAULT_GAMMA = 0.2   # new vulnerability class bonus weight


# ──────────────────────────────────────────────
# Agent Memory Manager
# ──────────────────────────────────────────────

class AgentMemoryManager:
    """
    Manages cross-episode agent memory for the self-improving agent system.

    Responsibilities:
        - Record episode outcomes (missed vulns, false positives, reasoning failures)
        - Compute improvement reward R_improve across consecutive episodes
        - Track tool usage patterns for efficiency analysis
        - Provide memory summaries for agent observations
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        max_history: int = 100,
    ):
        self._memory = AgentMemory()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._max_history = max_history

    # ======================================================================
    # Public API
    # ======================================================================

    @property
    def memory(self) -> AgentMemory:
        """Return the current memory state."""
        return self._memory

    def record_episode(
        self,
        task_id: str,
        grade_result: GradeResult,
        issues_detected: list[ReviewComment],
        tool_call_log: list[dict[str, Any]],
    ) -> None:
        """
        Record the outcome of a completed episode into memory.

        Updates missed vulnerabilities, false positives, tool stats,
        and the improvement trajectory.
        """
        episode_num = self._memory.total_episodes + 1
        self._memory.total_episodes = episode_num

        # ── Record missed vulnerabilities ──
        for missed in grade_result.missed_issues:
            self._memory.missed_vulnerabilities.append(
                VulnerabilityRecord(
                    episode_task_id=task_id,
                    issue_type=missed.issue,
                    severity=missed.severity,
                    line=missed.line,
                    explanation=missed.explanation,
                    episode_number=episode_num,
                )
            )

        # ── Record false positives ──
        for fp in grade_result.false_positives:
            self._memory.false_positives.append(
                FalsePositiveRecord(
                    episode_task_id=task_id,
                    issue_claimed=fp.issue,
                    line=fp.line,
                    episode_number=episode_num,
                )
            )

        # ── Record reasoning failures ──
        # A reasoning failure is when the agent found issues but they were
        # all false positives (poor reasoning despite effort)
        if issues_detected and not grade_result.matches:
            self._memory.reasoning_failures.append(
                ReasoningFailure(
                    episode_task_id=task_id,
                    description=(
                        f"Submitted {len(issues_detected)} findings but none matched "
                        f"ground truth ({len(grade_result.false_positives)} false positives)"
                    ),
                    phase=Phase.REFINEMENT,
                    episode_number=episode_num,
                )
            )

        # ── Update tool stats ──
        self._update_tool_stats(tool_call_log, grade_result)

        # ── Record episode score ──
        self._memory.improvement_trajectory.append(
            EpisodeScore(
                episode_number=episode_num,
                task_id=task_id,
                score=grade_result.total_score,
                precision=grade_result.precision,
                recall=grade_result.recall,
                f1=grade_result.f1,
                issues_found=len(grade_result.matches),
                false_positives=len(grade_result.false_positives),
                tool_calls_made=len(tool_call_log),
            )
        )

        # ── Trim history if needed ──
        self._trim_history()

        logger.info(
            "Episode %d recorded — score=%.4f, missed=%d, FP=%d",
            episode_num,
            grade_result.total_score,
            len(grade_result.missed_issues),
            len(grade_result.false_positives),
        )

    def calculate_improvement_reward(
        self,
        current_recall: float,
        current_precision: float,
        current_grade: GradeResult,
    ) -> ImprovementReward:
        """
        Compute the improvement reward R_improve across episodes.

        R_improve = α × (recall_t - recall_{t-1})
                  - β × repeated_mistakes
                  + γ × new_class_coverage
        """
        trajectory = self._memory.improvement_trajectory

        # Need at least one previous episode
        if not trajectory:
            return ImprovementReward(
                recall_gain=0.0,
                repeated_mistakes=0.0,
                new_class_coverage=0.0,
                total=0.0,
            )

        # ── Recall gain ──
        prev_recall = trajectory[-1].recall
        recall_gain = current_recall - prev_recall

        # ── Repeated mistakes ──
        # Count how many of the current missed issues were also missed before
        repeated = 0
        missed_types = {m.issue for m in current_grade.missed_issues}
        historical_misses = {v.issue_type for v in self._memory.missed_vulnerabilities}
        repeated = len(missed_types & historical_misses)
        repeated_penalty = repeated * 0.1  # 0.1 per repeat

        # ── New class coverage ──
        # Bonus for finding vulnerability types never found before
        found_types = {m.ground_truth.issue for m in current_grade.matches if m.is_match}
        historically_found = set()
        for ep in trajectory:
            # We can't perfectly reconstruct, so use missed type complement
            pass
        # Simpler: bonus for each found type that appears in missed history
        recovered = found_types & historical_misses
        new_class_bonus = len(recovered) * 0.15  # 0.15 per recovered class

        # ── Total ──
        total = (
            self._alpha * recall_gain
            - self._beta * repeated_penalty
            + self._gamma * new_class_bonus
        )

        return ImprovementReward(
            recall_gain=round(recall_gain, 4),
            repeated_mistakes=round(repeated_penalty, 4),
            new_class_coverage=round(new_class_bonus, 4),
            total=round(total, 4),
        )

    def get_memory_summary(self) -> dict[str, Any]:
        """
        Return a compact summary of agent memory for inclusion in observations.

        This gives the agent awareness of its historical weaknesses without
        exposing raw memory internals.
        """
        mem = self._memory

        # Aggregate missed vulnerability types
        missed_by_type: dict[str, int] = {}
        for v in mem.missed_vulnerabilities[-50:]:  # Last 50
            missed_by_type[v.issue_type] = missed_by_type.get(v.issue_type, 0) + 1

        # Aggregate false positive patterns
        fp_by_type: dict[str, int] = {}
        for fp in mem.false_positives[-50:]:
            fp_by_type[fp.issue_claimed] = fp_by_type.get(fp.issue_claimed, 0) + 1

        # Recent scores
        recent_scores = [
            {"task": e.task_id, "score": e.score, "recall": e.recall}
            for e in mem.improvement_trajectory[-5:]
        ]

        # Tool efficiency
        tool_summary = {}
        for tool_name, stats in mem.tool_usage_patterns.items():
            efficiency = (
                stats.calls_leading_to_findings / stats.total_calls
                if stats.total_calls > 0
                else 0.0
            )
            tool_summary[tool_name] = {
                "total_calls": stats.total_calls,
                "efficiency": round(efficiency, 3),
            }

        return {
            "total_episodes": mem.total_episodes,
            "commonly_missed_issues": dict(
                sorted(missed_by_type.items(), key=lambda x: -x[1])[:5]
            ),
            "common_false_positives": dict(
                sorted(fp_by_type.items(), key=lambda x: -x[1])[:5]
            ),
            "recent_scores": recent_scores,
            "tool_efficiency": tool_summary,
            "reasoning_failure_count": len(mem.reasoning_failures),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregated training metrics for the /metrics endpoint."""
        trajectory = self._memory.improvement_trajectory
        if not trajectory:
            return {
                "total_episodes": 0,
                "average_score": 0.0,
                "average_recall": 0.0,
                "average_precision": 0.0,
                "improvement_trend": 0.0,
                "total_missed_vulnerabilities": 0,
                "total_false_positives": 0,
                "tool_usage": {},
            }

        scores = [e.score for e in trajectory]
        recalls = [e.recall for e in trajectory]
        precisions = [e.precision for e in trajectory]

        # Compute improvement trend (slope of last 10 scores)
        recent = scores[-10:]
        if len(recent) >= 2:
            # Simple linear regression slope
            n = len(recent)
            x_mean = (n - 1) / 2
            y_mean = sum(recent) / n
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            trend = numerator / denominator if denominator > 0 else 0.0
        else:
            trend = 0.0

        return {
            "total_episodes": len(trajectory),
            "average_score": round(sum(scores) / len(scores), 4),
            "average_recall": round(sum(recalls) / len(recalls), 4),
            "average_precision": round(sum(precisions) / len(precisions), 4),
            "improvement_trend": round(trend, 6),
            "total_missed_vulnerabilities": len(self._memory.missed_vulnerabilities),
            "total_false_positives": len(self._memory.false_positives),
            "tool_usage": {
                name: stats.model_dump()
                for name, stats in self._memory.tool_usage_patterns.items()
            },
            "score_history": [
                {"episode": e.episode_number, "score": e.score, "task": e.task_id}
                for e in trajectory[-20:]
            ],
        }

    def reset(self) -> None:
        """Reset all memory (for testing or fresh starts)."""
        self._memory = AgentMemory()

    # ======================================================================
    # Internal
    # ======================================================================

    def _update_tool_stats(
        self,
        tool_call_log: list[dict[str, Any]],
        grade_result: GradeResult,
    ) -> None:
        """Update tool usage statistics from episode's tool calls."""
        seen_calls: set[str] = set()
        found_issue_types = {
            m.ground_truth.issue for m in grade_result.matches if m.is_match
        }

        for call in tool_call_log:
            tool_name = call.get("tool", "unknown")
            argument = call.get("argument", "")
            success = call.get("success", False)
            call_key = f"{tool_name}:{argument}"

            if tool_name not in self._memory.tool_usage_patterns:
                self._memory.tool_usage_patterns[tool_name] = ToolStats()

            stats = self._memory.tool_usage_patterns[tool_name]
            stats.total_calls += 1

            if success:
                stats.successful_calls += 1

            if call_key in seen_calls:
                stats.redundant_calls += 1
            seen_calls.add(call_key)

        # Heuristic: if tool was used and issues were found, credit the tools
        if tool_call_log and found_issue_types:
            for call in tool_call_log:
                tool_name = call.get("tool", "unknown")
                if tool_name in self._memory.tool_usage_patterns:
                    self._memory.tool_usage_patterns[tool_name].calls_leading_to_findings += 1

    def _trim_history(self) -> None:
        """Keep memory bounded to max_history entries per category."""
        m = self._memory
        if len(m.missed_vulnerabilities) > self._max_history:
            m.missed_vulnerabilities = m.missed_vulnerabilities[-self._max_history:]
        if len(m.false_positives) > self._max_history:
            m.false_positives = m.false_positives[-self._max_history:]
        if len(m.reasoning_failures) > self._max_history:
            m.reasoning_failures = m.reasoning_failures[-self._max_history:]
        if len(m.improvement_trajectory) > self._max_history:
            m.improvement_trajectory = m.improvement_trajectory[-self._max_history:]
