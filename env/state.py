"""
CodeRL State Models — Pydantic models for the OpenEnv-compliant RL environment.

Defines typed schemas for Observation, Action, Reward, State, and StepResult
used throughout the environment lifecycle.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Severity(str, Enum):
    """Issue severity levels, ordered by impact."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Difficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Phase(str, Enum):
    """Multi-phase reasoning stages within an episode."""
    SURFACE = "surface"
    LOGIC = "logic"
    SECURITY = "security"
    REFINEMENT = "refinement"


# ──────────────────────────────────────────────
# Ground Truth Models
# ──────────────────────────────────────────────

class GroundTruthIssue(BaseModel):
    """A known issue in a task, used as ground truth for grading."""
    line: int = Field(..., description="Line number where the issue occurs")
    issue: str = Field(..., description="Short issue identifier (e.g., 'SQL Injection')")
    severity: Severity = Field(..., description="Issue severity level")
    explanation: str = Field(..., description="Detailed explanation of why this is an issue")
    suggestion: str = Field(..., description="Recommended fix")


# ──────────────────────────────────────────────
# Task Model
# ──────────────────────────────────────────────

class Task(BaseModel):
    """A code review task with code diff and ground truth issues."""
    id: str = Field(..., description="Unique task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty level")
    file_name: str = Field(..., description="File being reviewed")
    code_diff: str = Field(..., description="The code diff to review")
    context: str = Field(..., description="Context about the code's purpose")
    language: str = Field(default="python", description="Programming language")
    ground_truth: list[GroundTruthIssue] = Field(..., description="Known issues for grading")
    # Optional: cross-file context for hard tasks
    related_files: Optional[dict[str, str]] = Field(
        default=None,
        description="Related file contents for cross-file reasoning"
    )
    # Optional: function/variable metadata for tool simulation
    function_signatures: Optional[dict[str, str]] = Field(
        default=None,
        description="Function name → signature mapping for inspect_function tool"
    )
    variable_traces: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Variable name → usage locations for trace_variable tool"
    )

# ──────────────────────────────────────────────
# Agent Memory Models (cross-episode)
# ──────────────────────────────────────────────

class VulnerabilityRecord(BaseModel):
    """A vulnerability the agent failed to detect in a past episode."""
    episode_task_id: str
    issue_type: str
    severity: Severity
    line: int
    explanation: str
    episode_number: int


class FalsePositiveRecord(BaseModel):
    """A false positive the agent submitted in a past episode."""
    episode_task_id: str
    issue_claimed: str
    line: int
    episode_number: int


class ReasoningFailure(BaseModel):
    """A breakdown in the agent's reasoning chain."""
    episode_task_id: str
    description: str
    phase: Phase
    episode_number: int


class ToolStats(BaseModel):
    """Usage statistics for a single tool."""
    total_calls: int = 0
    successful_calls: int = 0
    calls_leading_to_findings: int = 0
    redundant_calls: int = 0


class EpisodeScore(BaseModel):
    """Score record for a single episode."""
    episode_number: int
    task_id: str
    score: float
    precision: float
    recall: float
    f1: float
    issues_found: int
    false_positives: int
    tool_calls_made: int


class AgentMemory(BaseModel):
    """Cross-episode memory persisted across the agent's lifetime."""
    missed_vulnerabilities: list[VulnerabilityRecord] = Field(
        default_factory=list,
        description="Vulnerabilities the agent failed to detect"
    )
    false_positives: list[FalsePositiveRecord] = Field(
        default_factory=list,
        description="False positives the agent submitted"
    )
    reasoning_failures: list[ReasoningFailure] = Field(
        default_factory=list,
        description="Points where the agent's reasoning broke down"
    )
    tool_usage_patterns: dict[str, ToolStats] = Field(
        default_factory=dict,
        description="Per-tool usage statistics"
    )
    improvement_trajectory: list[EpisodeScore] = Field(
        default_factory=list,
        description="Score history across episodes"
    )
    total_episodes: int = Field(default=0, description="Total episodes completed")


# ──────────────────────────────────────────────
# Observation (returned to agent)
# ──────────────────────────────────────────────

class HistoryEntry(BaseModel):
    """A single step in the agent's interaction history."""
    step: int
    action_summary: str = Field(..., description="Brief summary of what the agent did")
    reward: float
    issues_found: int


class Observation(BaseModel):
    """What the agent sees at each step."""
    code_diff: str = Field(..., description="The code diff to review")
    file_name: str = Field(..., description="Name of the file being reviewed")
    context: str = Field(..., description="Purpose/context of the code")
    language: str = Field(default="python", description="Programming language")
    step: int = Field(..., description="Current step number (1-indexed)")
    max_steps: int = Field(..., description="Maximum allowed steps")
    phase: Phase = Field(default=Phase.SURFACE, description="Current reasoning phase")
    history: list[HistoryEntry] = Field(default_factory=list, description="Previous interactions")
    available_tools: list[str] = Field(
        default_factory=lambda: [
            "inspect_function", "trace_variable", "get_call_graph",
            "check_test_coverage", "inspect_import", "search_codebase",
        ],
        description="Tools the agent can invoke"
    )
    related_files: Optional[dict[str, str]] = Field(
        default=None,
        description="Related file contents (if available)"
    )
    task_id: str = Field(..., description="Current task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty")
    agent_memory_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="Summary of agent's cross-episode memory (past mistakes, patterns)"
    )
    phase_transition: Optional[str] = Field(
        default=None,
        description="Phase change notification, e.g. 'logic → security'"
    )


# ──────────────────────────────────────────────
# Action (agent's response)
# ──────────────────────────────────────────────

class ReviewComment(BaseModel):
    """A single code review comment from the agent."""
    line: int = Field(..., description="Line number of the issue")
    issue: str = Field(..., description="Short issue name")
    severity: Severity = Field(..., description="Issue severity")
    explanation: str = Field(..., description="Why this is a problem")
    suggestion: str = Field(..., description="How to fix it")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence [0-1]")


class ToolCall(BaseModel):
    """A tool invocation by the agent."""
    tool: str = Field(..., description="Tool name: inspect_function | trace_variable")
    argument: str = Field(..., description="Argument to pass to the tool")


class Action(BaseModel):
    """Agent's action: review comments and/or tool calls."""
    comments: list[ReviewComment] = Field(
        default_factory=list,
        description="Code review comments"
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations for deeper analysis"
    )


# ──────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────

class TrajectoryReward(BaseModel):
    """Trajectory-based reward computed over the full action sequence."""
    reasoning_consistency: float = Field(default=0.0, description="Coherence of findings across steps")
    efficient_reasoning: float = Field(default=0.0, description="Efficiency of reaching conclusions")
    redundant_actions: float = Field(default=0.0, description="Penalty for revisiting same areas")
    noisy_exploration: float = Field(default=0.0, description="Penalty for erratic tool usage")
    total: float = Field(default=0.0, description="Net trajectory reward")


class ImprovementReward(BaseModel):
    """Improvement reward computed across consecutive episodes."""
    recall_gain: float = Field(default=0.0, description="Recall improvement over last episode")
    repeated_mistakes: float = Field(default=0.0, description="Penalty for repeating known errors")
    new_class_coverage: float = Field(default=0.0, description="Bonus for detecting new vuln types")
    total: float = Field(default=0.0, description="Net improvement reward")


class ToolEfficiencyReward(BaseModel):
    """Tool usage efficiency reward."""
    information_gain: float = Field(default=0.0, description="Value from tool calls")
    redundant_call_penalty: float = Field(default=0.0, description="Penalty for redundant calls")
    total: float = Field(default=0.0, description="Net tool efficiency reward")


class RewardBreakdown(BaseModel):
    """Detailed breakdown of the four-component reward calculation."""
    # Step-level reward components
    precision_score: float = Field(..., description="Fraction of agent's issues that were correct")
    recall_score: float = Field(..., description="Fraction of ground truth issues found")
    severity_bonus: float = Field(..., description="Bonus for finding critical/high severity issues")
    false_positive_penalty: float = Field(..., description="Penalty for incorrect issues")
    duplicate_penalty: float = Field(..., description="Penalty for duplicate findings")
    step_reward: float = Field(default=0.0, description="Step-level reward subtotal")
    # Trajectory reward
    trajectory_reward: TrajectoryReward = Field(
        default_factory=TrajectoryReward,
        description="Trajectory-based reward"
    )
    # Improvement reward
    improvement_reward: ImprovementReward = Field(
        default_factory=ImprovementReward,
        description="Cross-episode improvement reward"
    )
    # Tool efficiency reward
    tool_efficiency_reward: ToolEfficiencyReward = Field(
        default_factory=ToolEfficiencyReward,
        description="Tool usage efficiency reward"
    )
    total: float = Field(..., description="Final composite reward (R_step + R_traj + R_improve + R_tool)")


# ──────────────────────────────────────────────
# State (internal)
# ──────────────────────────────────────────────

class State(BaseModel):
    """Full internal state of the environment."""
    task_id: str = Field(..., description="Current task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty")
    current_step: int = Field(default=0, description="Current step (0 = not started)")
    max_steps: int = Field(default=6, description="Maximum steps allowed")
    phase: Phase = Field(default=Phase.SURFACE, description="Current reasoning phase")
    issues_detected: list[ReviewComment] = Field(
        default_factory=list,
        description="All issues found so far"
    )
    history: list[HistoryEntry] = Field(
        default_factory=list,
        description="Interaction history"
    )
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated")
    done: bool = Field(default=False, description="Whether episode is finished")
    final_score: Optional[float] = Field(default=None, description="Final grader score [0-1]")
    tool_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Cached results from tool calls"
    )
    tool_call_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log of all tool calls made during the episode"
    )


# ──────────────────────────────────────────────
# Step Result (returned from env.step())
# ──────────────────────────────────────────────

class StepResult(BaseModel):
    """Result of a single environment step."""
    observation: Observation
    reward: RewardBreakdown
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
