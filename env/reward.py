"""
CodeRL Reward Calculator — Four-component dense reward system for code review.

Components:
    R_step     — Per-step precision/recall/severity rewards
    R_traj     — Trajectory-based reasoning quality (end of episode)
    R_improve  — Cross-episode improvement signal
    R_tool     — Tool usage efficiency

Total: R_total = R_step + R_traj + R_improve + R_tool

All weights configurable via config/reward.yaml.
"""

from __future__ import annotations

from typing import Any, Optional

from env.grader import Grader, GradeResult, SEVERITY_WEIGHTS
from env.state import (
    GroundTruthIssue,
    HistoryEntry,
    ImprovementReward,
    RewardBreakdown,
    ReviewComment,
    Severity,
    ToolEfficiencyReward,
    TrajectoryReward,
)


# ──────────────────────────────────────────────
# Reward Constants (defaults, overridable via YAML)
# ──────────────────────────────────────────────

PRECISION_WEIGHT = 0.5
RECALL_WEIGHT = 0.3
SEVERITY_BONUS_WEIGHT = 0.2

FALSE_POSITIVE_PENALTY = 0.5
DUPLICATE_PENALTY = 0.2

# Bonus for finding critical/high severity issues
SEVERITY_BONUS_MAP: dict[Severity, float] = {
    Severity.CRITICAL: 0.3,
    Severity.HIGH: 0.15,
    Severity.MEDIUM: 0.05,
    Severity.LOW: 0.0,
}

# Trajectory reward weights
TRAJECTORY_WEIGHTS = {
    "reasoning_consistency": 0.15,
    "efficient_reasoning": 0.10,
    "redundant_actions": 0.10,
    "noisy_exploration": 0.05,
}

# Tool efficiency weights
TOOL_EFFICIENCY_WEIGHTS = {
    "information_gain": 0.1,
    "redundant_call_penalty": 0.15,
}


# ──────────────────────────────────────────────
# Reward Calculator
# ──────────────────────────────────────────────

class RewardCalculator:
    """
    Calculates the four-component reward for code review actions.

    R_total = R_step + R_traj + R_improve + R_tool
    """

    def __init__(self, grader: Grader | None = None):
        self.grader = grader or Grader()

    def calculate(
        self,
        new_comments: list[ReviewComment],
        all_comments: list[ReviewComment],
        ground_truth: list[GroundTruthIssue],
        history: Optional[list[HistoryEntry]] = None,
        tool_call_log: Optional[list[dict[str, Any]]] = None,
        improvement_reward: Optional[ImprovementReward] = None,
        is_final_step: bool = False,
    ) -> RewardBreakdown:
        """
        Calculate the full reward for a step.

        Args:
            new_comments: Comments submitted in this step only
            all_comments: All comments accumulated so far (including new ones)
            ground_truth: Ground truth issues for the current task
            history: Full interaction history for trajectory reward
            tool_call_log: All tool calls for tool efficiency reward
            improvement_reward: Pre-computed improvement reward from memory manager
            is_final_step: Whether this is the last step of the episode

        Returns:
            RewardBreakdown with all four component scores and total
        """
        # ── R_step: Step-level reward ──
        step_reward = self._calculate_step_reward(
            new_comments, all_comments, ground_truth
        )

        # ── R_traj: Trajectory reward (only on final step) ──
        traj_reward = TrajectoryReward()
        if is_final_step and history:
            traj_reward = self._calculate_trajectory_reward(
                history, all_comments, tool_call_log or []
            )

        # ── R_improve: Improvement reward (passed in from memory manager) ──
        improve_reward = improvement_reward or ImprovementReward()

        # ── R_tool: Tool efficiency reward ──
        tool_reward = ToolEfficiencyReward()
        if tool_call_log:
            tool_reward = self._calculate_tool_efficiency(
                tool_call_log, new_comments, all_comments, ground_truth
            )

        # ── Total ──
        total = (
            step_reward["total"]
            + traj_reward.total
            + improve_reward.total
            + tool_reward.total
        )

        return RewardBreakdown(
            precision_score=step_reward["precision_score"],
            recall_score=step_reward["recall_score"],
            severity_bonus=step_reward["severity_bonus"],
            false_positive_penalty=step_reward["false_positive_penalty"],
            duplicate_penalty=step_reward["duplicate_penalty"],
            step_reward=round(step_reward["total"], 4),
            trajectory_reward=traj_reward,
            improvement_reward=improve_reward,
            tool_efficiency_reward=tool_reward,
            total=round(total, 4),
        )

    # ======================================================================
    # R_step: Step-level reward
    # ======================================================================

    def _calculate_step_reward(
        self,
        new_comments: list[ReviewComment],
        all_comments: list[ReviewComment],
        ground_truth: list[GroundTruthIssue],
    ) -> dict[str, float]:
        """Calculate the per-step reward (precision, recall, severity, penalties)."""
        if not new_comments:
            return {
                "precision_score": 0.0,
                "recall_score": 0.0,
                "severity_bonus": 0.0,
                "false_positive_penalty": 0.0,
                "duplicate_penalty": 0.0,
                "total": 0.0,
            }

        # Grade the cumulative set of comments
        grade = self.grader.grade(all_comments, ground_truth)

        # Precision
        precision_score = grade.precision * PRECISION_WEIGHT

        # Recall
        recall_score = grade.recall * RECALL_WEIGHT

        # Severity bonus
        severity_bonus = self._calculate_severity_bonus(grade)

        # False positive penalty
        fp_penalty = len(grade.false_positives) * FALSE_POSITIVE_PENALTY

        # Duplicate penalty
        dup_penalty = self._calculate_duplicate_penalty(new_comments, all_comments)

        total = (
            precision_score
            + recall_score
            + severity_bonus * SEVERITY_BONUS_WEIGHT
            - fp_penalty
            - dup_penalty
        )

        return {
            "precision_score": round(precision_score, 4),
            "recall_score": round(recall_score, 4),
            "severity_bonus": round(severity_bonus, 4),
            "false_positive_penalty": round(fp_penalty, 4),
            "duplicate_penalty": round(dup_penalty, 4),
            "total": round(total, 4),
        }

    # ======================================================================
    # R_traj: Trajectory-based reward
    # ======================================================================

    def _calculate_trajectory_reward(
        self,
        history: list[HistoryEntry],
        all_comments: list[ReviewComment],
        tool_call_log: list[dict[str, Any]],
    ) -> TrajectoryReward:
        """
        Compute trajectory reward over the full action sequence.

        R_traj = +w6 × reasoning_consistency
               + w7 × efficient_reasoning
               - w8 × redundant_actions
               - w9 × noisy_exploration
        """
        w = TRAJECTORY_WEIGHTS

        # ── Reasoning consistency ──
        # Reward if findings gradually accumulated (not all dumped on step 1)
        steps_with_findings = sum(1 for h in history if h.issues_found > 0)
        total_steps = len(history)
        # Ideal: findings spread across multiple steps
        consistency = steps_with_findings / total_steps if total_steps > 0 else 0.0

        # ── Efficient reasoning ──
        # Penalize steps with zero findings AND zero tool calls
        productive_steps = sum(
            1 for h in history if h.issues_found > 0 or "tool" in h.action_summary.lower()
        )
        efficiency = productive_steps / total_steps if total_steps > 0 else 0.0

        # ── Redundant actions ──
        # Count how many comments target the same line (within ±1)
        lines_seen: set[int] = set()
        redundant_count = 0
        for c in all_comments:
            if any(abs(c.line - seen) <= 1 for seen in lines_seen):
                redundant_count += 1
            lines_seen.add(c.line)
        redundant_ratio = redundant_count / len(all_comments) if all_comments else 0.0

        # ── Noisy exploration ──
        # Penalize calling many different tools without follow-up findings
        unique_tool_calls = len(set(
            f"{tc.get('tool')}:{tc.get('argument')}" for tc in tool_call_log
        )) if tool_call_log else 0
        findings_count = len(all_comments)
        # Noisy if many tool calls but few findings
        if unique_tool_calls > 0 and findings_count > 0:
            tool_to_finding_ratio = findings_count / unique_tool_calls
            noisy = max(0.0, 1.0 - tool_to_finding_ratio)
        elif unique_tool_calls > 0 and findings_count == 0:
            noisy = 1.0
        else:
            noisy = 0.0

        total = (
            w["reasoning_consistency"] * consistency
            + w["efficient_reasoning"] * efficiency
            - w["redundant_actions"] * redundant_ratio
            - w["noisy_exploration"] * noisy
        )

        return TrajectoryReward(
            reasoning_consistency=round(consistency, 4),
            efficient_reasoning=round(efficiency, 4),
            redundant_actions=round(redundant_ratio, 4),
            noisy_exploration=round(noisy, 4),
            total=round(total, 4),
        )

    # ======================================================================
    # R_tool: Tool efficiency reward
    # ======================================================================

    def _calculate_tool_efficiency(
        self,
        tool_call_log: list[dict[str, Any]],
        new_comments: list[ReviewComment],
        all_comments: list[ReviewComment],
        ground_truth: list[GroundTruthIssue],
    ) -> ToolEfficiencyReward:
        """
        Compute tool usage efficiency.

        R_tool = +w10 × information_gain
               - w11 × redundant_call_penalty
        """
        w = TOOL_EFFICIENCY_WEIGHTS

        if not tool_call_log:
            return ToolEfficiencyReward()

        # ── Information gain ──
        # If tool was called and new comments were submitted in the same step,
        # credit the tool with information gain
        successful_calls = sum(1 for tc in tool_call_log if tc.get("success", False))
        info_gain = (
            successful_calls / len(tool_call_log) if tool_call_log else 0.0
        )
        # Boost if findings followed tool calls
        if new_comments and successful_calls > 0:
            info_gain = min(1.0, info_gain + 0.2)

        # ── Redundant calls ──
        seen: set[str] = set()
        redundant = 0
        for tc in tool_call_log:
            key = f"{tc.get('tool')}:{tc.get('argument')}"
            if key in seen:
                redundant += 1
            seen.add(key)

        redundant_penalty = redundant * w["redundant_call_penalty"]

        total = w["information_gain"] * info_gain - redundant_penalty

        return ToolEfficiencyReward(
            information_gain=round(info_gain, 4),
            redundant_call_penalty=round(redundant_penalty, 4),
            total=round(total, 4),
        )

    # ======================================================================
    # Shared helpers
    # ======================================================================

    def _calculate_severity_bonus(self, grade: GradeResult) -> float:
        """Bonus for finding critical/high severity issues."""
        bonus = 0.0
        for match in grade.matches:
            if match.is_match:
                sev = match.ground_truth.severity
                bonus += SEVERITY_BONUS_MAP.get(sev, 0.0)
        return bonus

    def _calculate_duplicate_penalty(
        self,
        new_comments: list[ReviewComment],
        all_comments: list[ReviewComment],
    ) -> float:
        """
        Penalize duplicate findings in the new batch.

        A comment is a duplicate if a previously submitted comment
        covers the same line AND has a similar issue name.
        """
        previous = all_comments[: len(all_comments) - len(new_comments)]
        if not previous:
            return 0.0

        duplicates = 0
        for new in new_comments:
            for old in previous:
                if (
                    abs(new.line - old.line) <= 1
                    and new.issue.lower().strip() == old.issue.lower().strip()
                ):
                    duplicates += 1
                    break

        return duplicates * DUPLICATE_PENALTY
