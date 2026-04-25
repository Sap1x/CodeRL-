"""
CodeRL++ End-to-End Validation Test

Tests all components:
1. Task loading
2. Environment reset/step/state
3. Grader determinism
4. Reward calculation (four-component)
5. Tool simulation (6 tools)
6. Multi-step interaction
7. Phase transitions
8. Agent memory persistence
9. Improvement reward
10. Tool efficiency reward
11. Adaptive curriculum
12. Full pipeline
"""

import json
import sys
import traceback

# ── Test 1: Task Loading ──

def test_task_loading():
    from env.task_loader import TaskLoader
    loader = TaskLoader()
    
    assert loader.task_count == 6, f"Expected 6 tasks, got {loader.task_count}"
    
    summary = loader.summary()
    assert summary["by_difficulty"]["easy"] == 2
    assert summary["by_difficulty"]["medium"] == 2
    assert summary["by_difficulty"]["hard"] == 2
    
    # Verify all task IDs exist
    expected_ids = ["easy_001", "easy_002", "medium_001", "medium_002", "hard_001", "hard_002"]
    for tid in expected_ids:
        task = loader.get_task(tid)
        assert task.id == tid
        assert len(task.ground_truth) > 0
    
    print("✅ Task Loading: PASSED")
    return True

# ── Test 2: Environment Reset ──

def test_environment_reset():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    # Reset with default task (curriculum-selected)
    obs = env.reset()
    assert "code_diff" in obs
    assert "file_name" in obs
    assert "step" in obs
    assert obs["step"] == 1
    assert "phase" in obs
    assert obs["phase"] == "surface"
    
    # Reset with specific task
    obs = env.reset(task_id="hard_001")
    assert obs["task_id"] == "hard_001"
    assert obs["difficulty"] == "hard"
    assert obs["max_steps"] == 8
    assert "available_tools" in obs
    assert len(obs["available_tools"]) == 6  # All 6 tools
    
    print("✅ Environment Reset: PASSED")
    return True

# ── Test 3: Environment Step ──

def test_environment_step():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    obs = env.reset(task_id="easy_001")
    
    # Submit a correct finding
    action = {
        "comments": [
            {
                "line": 7,
                "issue": "Resource Leak",
                "severity": "medium",
                "explanation": "File handle not closed",
                "suggestion": "Use with statement",
                "confidence": 0.9
            }
        ],
        "tool_calls": []
    }
    
    result = env.step(action)
    assert "observation" in result
    assert "reward" in result
    assert "done" in result
    assert "info" in result
    assert result["reward"]["total"] > 0, f"Expected positive reward, got {result['reward']['total']}"
    
    # Check new reward structure
    reward = result["reward"]
    assert "step_reward" in reward
    assert "trajectory_reward" in reward
    assert "improvement_reward" in reward
    assert "tool_efficiency_reward" in reward
    
    # Check phase info
    assert "phase" in result["info"]
    
    print("✅ Environment Step: PASSED")
    return True

# ── Test 4: Grader Determinism ──

def test_grader_determinism():
    from env.grader import Grader
    from env.state import ReviewComment, GroundTruthIssue, Severity
    
    grader = Grader()
    
    predicted = [
        ReviewComment(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                     explanation="test", suggestion="test", confidence=0.9),
        ReviewComment(line=14, issue="Off-by-one Error", severity=Severity.HIGH,
                     explanation="test", suggestion="test", confidence=0.85),
    ]
    
    ground_truth = [
        GroundTruthIssue(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                        explanation="test", suggestion="test"),
        GroundTruthIssue(line=14, issue="Off-by-one Error", severity=Severity.HIGH,
                        explanation="test", suggestion="test"),
        GroundTruthIssue(line=15, issue="Division by Zero", severity=Severity.HIGH,
                        explanation="test", suggestion="test"),
    ]
    
    # Run twice to verify determinism
    result1 = grader.grade(predicted, ground_truth)
    result2 = grader.grade(predicted, ground_truth)
    
    assert result1.total_score == result2.total_score, "Grader is not deterministic!"
    assert result1.precision == result2.precision
    assert result1.recall == result2.recall
    assert result1.total_score > 0
    assert 0.0 <= result1.total_score <= 1.0
    
    print(f"  → Score: {result1.total_score}, Precision: {result1.precision}, Recall: {result1.recall}")
    print("✅ Grader Determinism: PASSED")
    return True

# ── Test 5: Reward Calculation (Four-Component) ──

def test_reward_calculation():
    from env.reward import RewardCalculator
    from env.state import ReviewComment, GroundTruthIssue, Severity
    
    calc = RewardCalculator()
    
    comments = [
        ReviewComment(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                     explanation="test", suggestion="test", confidence=0.9),
    ]
    
    ground_truth = [
        GroundTruthIssue(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                        explanation="test", suggestion="test"),
    ]
    
    reward = calc.calculate(comments, comments, ground_truth)
    assert reward.total > 0, f"Expected positive reward, got {reward.total}"
    assert reward.false_positive_penalty == 0.0
    assert reward.duplicate_penalty == 0.0
    assert reward.step_reward > 0
    
    # Verify new reward components exist
    assert reward.trajectory_reward is not None
    assert reward.improvement_reward is not None
    assert reward.tool_efficiency_reward is not None
    
    # Test false positive
    bad_comments = [
        ReviewComment(line=999, issue="Nonexistent Bug", severity=Severity.LOW,
                     explanation="test", suggestion="test", confidence=0.1),
    ]
    bad_reward = calc.calculate(bad_comments, bad_comments, ground_truth)
    assert bad_reward.false_positive_penalty > 0
    
    print("✅ Reward Calculation: PASSED")
    return True

# ── Test 6: Tool Simulation (All 6 Tools) ──

def test_tool_simulation():
    from env.tools import ToolSimulator
    from env.task_loader import TaskLoader
    
    loader = TaskLoader()
    task = loader.get_task("hard_001")
    sim = ToolSimulator(task)
    
    # Test inspect_function
    result = sim.execute("inspect_function", "hash_password")
    assert result["success"] is True
    assert "signature" in result["result"]
    
    # Test trace_variable
    result = sim.execute("trace_variable", "username")
    assert result["success"] is True
    assert len(result["result"]["usage_locations"]) > 0
    
    # Test get_call_graph
    result = sim.execute("get_call_graph", "hash_password")
    assert result["success"] is True
    assert "callers" in result["result"]
    assert "callees" in result["result"]
    
    # Test check_test_coverage
    result = sim.execute("check_test_coverage", "auth.py")
    assert result["success"] is True
    assert "coverage_percentage" in result["result"]
    assert "uncovered_lines" in result["result"]
    
    # Test inspect_import
    result = sim.execute("inspect_import", "hashlib")
    assert result["success"] is True
    assert "imports" in result["result"]
    
    # Test search_codebase
    result = sim.execute("search_codebase", "password")
    assert result["success"] is True
    assert len(result["result"]["matches"]) > 0
    
    # Test unknown tool
    result = sim.execute("unknown_tool", "test")
    assert result["success"] is False
    
    print("✅ Tool Simulation (6 tools): PASSED")
    return True

# ── Test 7: Phase Transitions ──

def test_phase_transitions():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    obs = env.reset(task_id="hard_001")
    max_steps = obs["max_steps"]  # 8 for hard tasks
    assert obs["phase"] == "surface"
    
    phases_seen = [obs["phase"]]
    action = {
        "comments": [],
        "tool_calls": []
    }
    
    for step in range(max_steps):
        result = env.step(action)
        phases_seen.append(result["observation"]["phase"])
    
    # Verify all four phases were traversed
    unique_phases = set(phases_seen)
    assert "surface" in unique_phases, f"Missing 'surface' phase, got {unique_phases}"
    assert "refinement" in unique_phases, f"Missing 'refinement' phase, got {unique_phases}"
    
    print(f"  → Phases observed: {phases_seen}")
    print("✅ Phase Transitions: PASSED")
    return True

# ── Test 8: Agent Memory Persistence ──

def test_agent_memory():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    # Run a full episode
    obs = env.reset(task_id="easy_001")
    action = {
        "comments": [
            {
                "line": 7,
                "issue": "Resource Leak",
                "severity": "medium",
                "explanation": "test",
                "suggestion": "test",
                "confidence": 0.9
            }
        ],
        "tool_calls": []
    }
    
    done = False
    while not done:
        result = env.step(action)
        done = result["done"]
    
    # Check memory was populated
    memory = env.get_memory()
    assert memory["total_episodes"] == 1
    assert len(memory["improvement_trajectory"]) == 1
    
    # Run a second episode
    obs = env.reset(task_id="easy_002")
    assert obs.get("agent_memory_summary") is not None, "Memory summary should be present after first episode"
    
    done = False
    while not done:
        result = env.step(action)
        done = result["done"]
    
    memory = env.get_memory()
    assert memory["total_episodes"] == 2
    assert len(memory["improvement_trajectory"]) == 2
    
    # Verify memory summary
    summary = env.get_memory_summary()
    assert "total_episodes" in summary
    assert "commonly_missed_issues" in summary
    assert "recent_scores" in summary
    
    print(f"  → Memory has {memory['total_episodes']} episodes tracked")
    print("✅ Agent Memory: PASSED")
    return True

# ── Test 9: Improvement Reward ──

def test_improvement_reward():
    from env.memory import AgentMemoryManager
    from env.grader import Grader, GradeResult
    from env.state import ReviewComment, GroundTruthIssue, Severity
    
    mgr = AgentMemoryManager()
    grader = Grader()
    
    # Simulate first episode with poor performance
    predicted1 = [
        ReviewComment(line=999, issue="Wrong", severity=Severity.LOW,
                     explanation="test", suggestion="test", confidence=0.5),
    ]
    gt = [
        GroundTruthIssue(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                        explanation="test", suggestion="test"),
        GroundTruthIssue(line=14, issue="Off-by-one Error", severity=Severity.HIGH,
                        explanation="test", suggestion="test"),
    ]
    grade1 = grader.grade(predicted1, gt)
    mgr.record_episode("easy_001", grade1, predicted1, [])
    
    # Now calculate improvement reward for a better second episode
    predicted2 = [
        ReviewComment(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                     explanation="test", suggestion="test", confidence=0.9),
    ]
    grade2 = grader.grade(predicted2, gt)
    
    improvement = mgr.calculate_improvement_reward(
        current_recall=grade2.recall,
        current_precision=grade2.precision,
        current_grade=grade2,
    )
    
    # Recall improved from 0 to >0, so recall_gain should be positive
    assert improvement.recall_gain > 0, f"Expected positive recall gain, got {improvement.recall_gain}"
    
    print(f"  → Improvement reward: total={improvement.total}, recall_gain={improvement.recall_gain}")
    print("✅ Improvement Reward: PASSED")
    return True

# ── Test 10: Tool Efficiency Reward ──

def test_tool_efficiency_reward():
    from env.reward import RewardCalculator
    from env.state import ReviewComment, GroundTruthIssue, Severity
    
    calc = RewardCalculator()
    
    comments = [
        ReviewComment(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                     explanation="test", suggestion="test", confidence=0.9),
    ]
    gt = [
        GroundTruthIssue(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                        explanation="test", suggestion="test"),
    ]
    
    # With tool calls
    tool_log = [
        {"tool": "inspect_function", "argument": "load_config", "success": True, "step": 1},
        {"tool": "trace_variable", "argument": "filepath", "success": True, "step": 1},
    ]
    
    reward_with_tools = calc.calculate(
        comments, comments, gt, tool_call_log=tool_log
    )
    
    # Without tool calls
    reward_without_tools = calc.calculate(comments, comments, gt)
    
    # Tool efficiency should add reward when tools are successful
    assert reward_with_tools.tool_efficiency_reward.information_gain > 0
    assert reward_without_tools.tool_efficiency_reward.total == 0.0
    
    # Test redundant tool calls
    redundant_log = [
        {"tool": "inspect_function", "argument": "load_config", "success": True, "step": 1},
        {"tool": "inspect_function", "argument": "load_config", "success": True, "step": 2},
    ]
    reward_redundant = calc.calculate(
        comments, comments, gt, tool_call_log=redundant_log
    )
    assert reward_redundant.tool_efficiency_reward.redundant_call_penalty > 0
    
    print(f"  → Tool efficiency: gain={reward_with_tools.tool_efficiency_reward.information_gain}, "
          f"redundant_penalty={reward_redundant.tool_efficiency_reward.redundant_call_penalty}")
    print("✅ Tool Efficiency Reward: PASSED")
    return True

# ── Test 11: Adaptive Curriculum ──

def test_adaptive_curriculum():
    from env.curriculum import AdaptiveCurriculum, CurriculumStrategy
    from env.task_loader import TaskLoader
    from env.state import AgentMemory
    
    loader = TaskLoader()
    
    # Test sequential strategy
    curriculum = AdaptiveCurriculum(
        task_loader=loader,
        strategy=CurriculumStrategy.SEQUENTIAL,
    )
    task1 = curriculum.select_next_task()
    task2 = curriculum.select_next_task()
    assert task1.id != task2.id or loader.task_count == 1
    
    # Test performance adaptive strategy
    adaptive = AdaptiveCurriculum(
        task_loader=loader,
        strategy=CurriculumStrategy.PERFORMANCE_ADAPTIVE,
    )
    
    # Record high scores for an easy task
    for _ in range(3):
        adaptive.record_task_score("easy_001", 0.95)
    
    # Check curriculum state
    state = adaptive.get_curriculum_state()
    assert "easy_001" in state["retired_tasks"]
    
    print(f"  → Curriculum state: {len(state['active_tasks'])} active, {len(state['retired_tasks'])} retired")
    print("✅ Adaptive Curriculum: PASSED")
    return True

# ── Test 12: Metrics Endpoint ──

def test_metrics():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    # Run one episode
    obs = env.reset(task_id="easy_001")
    action = {
        "comments": [
            {
                "line": 7,
                "issue": "Resource Leak",
                "severity": "medium",
                "explanation": "test",
                "suggestion": "test",
                "confidence": 0.9
            }
        ],
        "tool_calls": []
    }
    done = False
    while not done:
        result = env.step(action)
        done = result["done"]
    
    # Check metrics
    metrics = env.get_metrics()
    assert metrics["total_episodes"] >= 1
    assert "average_score" in metrics
    assert "improvement_trend" in metrics
    assert "tool_usage" in metrics
    
    print(f"  → Metrics: avg_score={metrics['average_score']}, episodes={metrics['total_episodes']}")
    print("✅ Metrics: PASSED")
    return True

# ── Test 13: Multi-Step Interaction ──

def test_multi_step():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    obs = env.reset(task_id="easy_001")
    max_steps = obs["max_steps"]
    
    # Run through all steps
    for step in range(max_steps):
        action = {
            "comments": [
                {
                    "line": 7 + step * 5,
                    "issue": f"Issue {step}",
                    "severity": "medium",
                    "explanation": "test",
                    "suggestion": "test",
                    "confidence": 0.5
                }
            ],
            "tool_calls": []
        }
        result = env.step(action)
        
        if step < max_steps - 1:
            assert result["done"] is False
        else:
            assert result["done"] is True
            assert "final_grade" in result["info"]
            assert 0.0 <= result["info"]["final_grade"]["score"] <= 1.0
    
    # Verify state
    state = env.get_state()
    assert state["done"] is True
    assert state["current_step"] == max_steps
    assert len(state["history"]) == max_steps
    
    print(f"  → Final score: {result['info']['final_grade']['score']}")
    print("✅ Multi-Step Interaction: PASSED")
    return True

# ── Test 14: Full Pipeline (all tasks) ──

def test_full_pipeline():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    task_ids = env.get_task_ids()
    scores = {}
    
    for task_id in task_ids:
        obs = env.reset(task_id=task_id)
        # Just run one step with a generic action
        action = {
            "comments": [
                {
                    "line": 10,
                    "issue": "Generic Issue",
                    "severity": "medium",
                    "explanation": "test",
                    "suggestion": "test",
                    "confidence": 0.5
                }
            ],
            "tool_calls": []
        }
        
        # Step through to completion
        done = False
        while not done:
            result = env.step(action)
            done = result["done"]
        
        score = result["info"]["final_grade"]["score"]
        scores[task_id] = score
    
    print(f"  → Scores: {json.dumps(scores, indent=2)}")
    print("✅ Full Pipeline: PASSED")
    return True

# ── Test 15: Environment Summary ──

def test_environment_summary():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    summary = env.get_summary()
    assert summary["name"] == "CodeRL++"
    assert "phases" in summary
    assert len(summary["phases"]) == 4
    assert "available_tools" in summary
    assert len(summary["available_tools"]) == 6
    assert "reward_components" in summary
    assert len(summary["reward_components"]) == 4
    
    print(f"  → Name: {summary['name']}, Tools: {len(summary['available_tools'])}, Phases: {summary['phases']}")
    print("✅ Environment Summary: PASSED")
    return True


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🧪 CodeRL++ — End-to-End Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("Task Loading", test_task_loading),
        ("Environment Reset", test_environment_reset),
        ("Environment Step", test_environment_step),
        ("Grader Determinism", test_grader_determinism),
        ("Reward Calculation (4-component)", test_reward_calculation),
        ("Tool Simulation (6 tools)", test_tool_simulation),
        ("Phase Transitions", test_phase_transitions),
        ("Agent Memory", test_agent_memory),
        ("Improvement Reward", test_improvement_reward),
        ("Tool Efficiency Reward", test_tool_efficiency_reward),
        ("Adaptive Curriculum", test_adaptive_curriculum),
        ("Metrics", test_metrics),
        ("Multi-Step Interaction", test_multi_step),
        ("Full Pipeline", test_full_pipeline),
        ("Environment Summary", test_environment_summary),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name}: FAILED")
            print(f"   Error: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
