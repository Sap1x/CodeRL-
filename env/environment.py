"""
CodeRL++ Environment — Production-grade OpenEnv-compliant RL environment
for agentic code review with self-improving capabilities.

Implements the core OpenEnv interface:
    - reset(task_id?) → Observation
    - step(action) → (Observation, Reward, done, info)
    - state() → State

Extended with:
    - Multi-phase reasoning (surface → logic → security → refinement)
    - Cross-episode agent memory
    - Four-component reward system (step + trajectory + improvement + tool)
    - Adaptive curriculum
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from env.curriculum import AdaptiveCurriculum, CurriculumStrategy
from env.grader import Grader, GradeResult
from env.memory import AgentMemoryManager
from env.reward import RewardCalculator
from env.state import (
    Action,
    Difficulty,
    HistoryEntry,
    Observation,
    Phase,
    RewardBreakdown,
    ReviewComment,
    State,
    StepResult,
    Task,
)
from env.task_loader import TaskLoader
from env.tools import ToolSimulator

logger = logging.getLogger("coderl.environment")


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DEFAULT_MAX_STEPS = 6

MAX_STEPS_BY_DIFFICULTY = {
    Difficulty.EASY: 5,
    Difficulty.MEDIUM: 6,
    Difficulty.HARD: 8,
}

# Phase boundaries as fractions of max_steps
PHASE_BOUNDARIES = [
    (0.00, 0.25, Phase.SURFACE),
    (0.25, 0.50, Phase.LOGIC),
    (0.50, 0.75, Phase.SECURITY),
    (0.75, 1.00, Phase.REFINEMENT),
]


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class CodeReviewEnv:
    """
    OpenEnv-compliant Reinforcement Learning environment for code review.

    The agent reviews code diffs, identifies issues, and receives
    dense rewards based on precision, recall, and severity matching.
    Supports multi-step reasoning with tool calls, cross-episode
    memory, and adaptive curriculum.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._task_loader = TaskLoader(data_dir=data_dir)
        self._grader = Grader()
        self._reward_calc = RewardCalculator(grader=self._grader)
        self._memory_mgr = AgentMemoryManager()
        self._curriculum = AdaptiveCurriculum(
            task_loader=self._task_loader,
            strategy=CurriculumStrategy.PERFORMANCE_ADAPTIVE,
        )
        self._tool_sim: Optional[ToolSimulator] = None

        # Current episode state
        self._current_task: Optional[Task] = None
        self._state: Optional[State] = None
        self._initialized = False
        self._previous_phase: Optional[Phase] = None

        logger.info(
            "CodeReviewEnv initialized — %d tasks loaded", self._task_loader.task_count
        )

    # ======================================================================
    # OpenEnv Interface
    # ======================================================================

    def reset(self, task_id: Optional[str] = None) -> dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Specific task to load. If None, uses adaptive curriculum.

        Returns:
            Initial observation as a dict.
        """
        if task_id:
            task = self._task_loader.get_task(task_id)
        else:
            # Use adaptive curriculum to select the next task
            task = self._curriculum.select_next_task(
                memory=self._memory_mgr.memory
            )

        max_steps = MAX_STEPS_BY_DIFFICULTY.get(task.difficulty, DEFAULT_MAX_STEPS)

        self._current_task = task
        self._tool_sim = ToolSimulator(task)
        self._previous_phase = None
        self._state = State(
            task_id=task.id,
            difficulty=task.difficulty,
            current_step=0,
            max_steps=max_steps,
            phase=Phase.SURFACE,
            issues_detected=[],
            history=[],
            cumulative_reward=0.0,
            done=False,
            final_score=None,
            tool_results={},
            tool_call_log=[],
        )
        self._initialized = True

        observation = self._build_observation()

        logger.info(
            "Environment reset — task=%s difficulty=%s max_steps=%d",
            task.id, task.difficulty.value, max_steps,
        )

        return observation.model_dump()

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Process an agent action and advance the environment.

        Args:
            action: Dict matching Action schema (comments + optional tool_calls)

        Returns:
            StepResult as a dict with observation, reward, done, info.
        """
        self._ensure_initialized()
        assert self._state is not None
        assert self._current_task is not None

        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Parse action
        parsed_action = Action(**action)

        # Advance step
        self._state.current_step += 1

        # ── Update phase ──
        old_phase = self._state.phase
        new_phase = self._compute_phase(self._state.current_step, self._state.max_steps)
        self._state.phase = new_phase
        phase_transition = (
            f"{old_phase.value} → {new_phase.value}" if old_phase != new_phase else None
        )

        # ── Process tool calls ──
        tool_results = []
        if parsed_action.tool_calls and self._tool_sim:
            for tc in parsed_action.tool_calls:
                result = self._tool_sim.execute(tc.tool, tc.argument)
                tool_results.append(result)
                self._state.tool_results[f"{tc.tool}:{tc.argument}"] = result
                self._state.tool_call_log.append({
                    "tool": tc.tool,
                    "argument": tc.argument,
                    "success": result.get("success", False),
                    "step": self._state.current_step,
                })

        # ── Process review comments ──
        new_comments = parsed_action.comments
        self._state.issues_detected.extend(new_comments)

        # ── Check if done ──
        done = self._state.current_step >= self._state.max_steps

        # ── Calculate improvement reward (cross-episode) ──
        improvement_reward = None
        if done:
            final_grade = self._grader.grade(
                self._state.issues_detected,
                self._current_task.ground_truth,
            )
            improvement_reward = self._memory_mgr.calculate_improvement_reward(
                current_recall=final_grade.recall,
                current_precision=final_grade.precision,
                current_grade=final_grade,
            )

        # ── Calculate reward ──
        reward = self._reward_calc.calculate(
            new_comments=new_comments,
            all_comments=self._state.issues_detected,
            ground_truth=self._current_task.ground_truth,
            history=list(self._state.history),
            tool_call_log=self._state.tool_call_log,
            improvement_reward=improvement_reward,
            is_final_step=done,
        )
        self._state.cumulative_reward += reward.total

        # ── Add history entry ──
        action_summary = self._summarize_action(parsed_action, tool_results)
        self._state.history.append(
            HistoryEntry(
                step=self._state.current_step,
                action_summary=action_summary,
                reward=reward.total,
                issues_found=len(new_comments),
            )
        )

        # ── Build info dict ──
        info: dict[str, Any] = {
            "step": self._state.current_step,
            "phase": new_phase.value,
            "phase_transition": phase_transition,
            "tool_results": tool_results,
            "new_issues_count": len(new_comments),
            "total_issues_found": len(self._state.issues_detected),
        }

        # ── Final grading if done ──
        if done:
            self._state.done = True
            # Use already computed final_grade if available, otherwise compute
            if improvement_reward is not None:
                # Already computed above
                pass
            else:
                final_grade = self._grader.grade(
                    self._state.issues_detected,
                    self._current_task.ground_truth,
                )

            self._state.final_score = final_grade.total_score
            info["final_grade"] = {
                "score": final_grade.total_score,
                "precision": final_grade.precision,
                "recall": final_grade.recall,
                "f1": final_grade.f1,
                "severity_weighted": final_grade.severity_weighted_score,
                "details": final_grade.details,
            }

            # ── Record episode in memory ──
            self._memory_mgr.record_episode(
                task_id=self._state.task_id,
                grade_result=final_grade,
                issues_detected=self._state.issues_detected,
                tool_call_log=self._state.tool_call_log,
            )

            # ── Update curriculum ──
            self._curriculum.record_task_score(
                self._state.task_id, final_grade.total_score
            )

            logger.info(
                "Episode complete — task=%s score=%.4f steps=%d phase=%s",
                self._state.task_id,
                final_grade.total_score,
                self._state.current_step,
                new_phase.value,
            )

        # ── Build observation ──
        observation = self._build_observation(phase_transition=phase_transition)

        step_result = StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

        return step_result.model_dump()

    def get_state(self) -> dict[str, Any]:
        """Return the current internal state."""
        self._ensure_initialized()
        assert self._state is not None
        return self._state.model_dump()

    # ======================================================================
    # Extended API (Memory, Metrics, Curriculum)
    # ======================================================================

    def get_memory(self) -> dict[str, Any]:
        """Return the current agent memory state."""
        return self._memory_mgr.memory.model_dump()

    def get_memory_summary(self) -> dict[str, Any]:
        """Return a compact memory summary."""
        return self._memory_mgr.get_memory_summary()

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregated training metrics."""
        return self._memory_mgr.get_metrics()

    def get_curriculum_state(self) -> dict[str, Any]:
        """Return the current curriculum state."""
        return self._curriculum.get_curriculum_state()

    # ======================================================================
    # Convenience Methods
    # ======================================================================

    def get_task_ids(self) -> list[str]:
        """Return all available task IDs."""
        return self._task_loader.get_all_task_ids()

    def get_summary(self) -> dict[str, Any]:
        """Return environment summary."""
        return {
            "name": "CodeRL++",
            "version": "1.0.0",
            "description": (
                "Self-Improving RL Environment for Agentic Code Review — "
                "Multi-phase reasoning with cross-episode memory and adaptive curriculum"
            ),
            "tasks": self._task_loader.summary(),
            "max_steps_by_difficulty": {
                k.value: v for k, v in MAX_STEPS_BY_DIFFICULTY.items()
            },
            "phases": [p.value for p in Phase],
            "available_tools": [
                "inspect_function", "trace_variable", "get_call_graph",
                "check_test_coverage", "inspect_import", "search_codebase",
            ],
            "reward_components": [
                "step_reward", "trajectory_reward",
                "improvement_reward", "tool_efficiency_reward",
            ],
            "curriculum_strategy": self._curriculum.get_curriculum_state()["strategy"],
            "total_episodes_completed": self._memory_mgr.memory.total_episodes,
        }

    # ======================================================================
    # Internal
    # ======================================================================

    def _build_observation(
        self,
        phase_transition: Optional[str] = None,
    ) -> Observation:
        """Build the current observation for the agent."""
        assert self._current_task is not None
        assert self._state is not None

        # Get memory summary for the agent (if we have history)
        memory_summary = None
        if self._memory_mgr.memory.total_episodes > 0:
            memory_summary = self._memory_mgr.get_memory_summary()

        return Observation(
            code_diff=self._current_task.code_diff,
            file_name=self._current_task.file_name,
            context=self._current_task.context,
            language=self._current_task.language,
            step=self._state.current_step + 1,  # 1-indexed for agent
            max_steps=self._state.max_steps,
            phase=self._state.phase,
            history=list(self._state.history),
            available_tools=[
                "inspect_function", "trace_variable", "get_call_graph",
                "check_test_coverage", "inspect_import", "search_codebase",
            ],
            related_files=self._current_task.related_files,
            task_id=self._current_task.id,
            difficulty=self._current_task.difficulty,
            agent_memory_summary=memory_summary,
            phase_transition=phase_transition,
        )

    def _compute_phase(self, current_step: int, max_steps: int) -> Phase:
        """Determine the reasoning phase based on step progress."""
        progress = current_step / max_steps if max_steps > 0 else 0.0
        for low, high, phase in PHASE_BOUNDARIES:
            if low <= progress < high:
                return phase
        return Phase.REFINEMENT

    def _summarize_action(
        self, action: Action, tool_results: list[dict]
    ) -> str:
        """Create a brief summary of the agent's action."""
        parts = []
        if action.comments:
            issues = [c.issue for c in action.comments]
            parts.append(f"Found {len(action.comments)} issue(s): {', '.join(issues)}")
        if action.tool_calls:
            tools = [f"{tc.tool}({tc.argument})" for tc in action.tool_calls]
            parts.append(f"Used tools: {', '.join(tools)}")
        return "; ".join(parts) if parts else "No action taken"

    def _ensure_initialized(self) -> None:
        """Ensure the environment has been reset."""
        if not self._initialized or self._state is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )
