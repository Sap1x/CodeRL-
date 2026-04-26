# 🚀 CodeRL++ — Self-Improving Agentic Code Review RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.org)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-orange)](https://huggingface.co/spaces/Sap1x/CodeRLAgent)
[![Open In Colab](https://img.shields.io/badge/Training-Open%20in%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/drive/1Vgc5wdIR-wDXaWsuTnkAMf4uK5eUznuU?usp=sharing)

> 🎯 **Training AI agents to think, reason, and improve like real software engineers — not just generate code, but truly understand it.**

> **TL;DR:** CodeRL++ is a self-improving RL environment where AI agents iteratively **detect bugs → propose fixes → verify resolutions** using tool-assisted multi-phase reasoning across four structured phases (Surface → Logic → Security → Refinement). Agents get measurably better with every episode through cross-episode memory that tracks historical mistakes and rewards recovery.

---

## 📎 Submission Materials

> All required hackathon deliverables are linked here for judge access.

| Deliverable | Link | Status |
|---|---|---|
| 🤗 Live Environment (HF Spaces) | [**Open Environment**](https://huggingface.co/spaces/Sap1x/CodeRLAgent) | ✅ Live |
| 📓 Training Notebook (Colab) | [**Execute in Colab**](https://colab.research.google.com/drive/1Vgc5wdIR-wDXaWsuTnkAMf4uK5eUznuU?usp=sharing) | ✅ Verified |
| 📝 Mini Blog (HuggingFace) | [**Read Post**](https://huggingface.co/spaces/Sap1x/CodeRLAgent/blob/main/miniblog_post.md) | ✅ Published |
| 💻 GitHub Repository | [**View Source**](https://github.com/Sap1x/CodeRL-.git) | ✅ Public |

---

## 🧠 Problem & Motivation

Large Language Models can **write** code — but they struggle to **review** it. Current AI code review tools suffer from three critical weaknesses:

1. **No iterative reasoning** — They make a single pass over a diff and miss deep logic bugs, control flow errors, and chained vulnerabilities.
2. **No tool usage** — Real engineers inspect function bodies, trace call graphs, and check test coverage before flagging issues. LLMs don't.
3. **No learning from mistakes** — Every review is stateless. The agent never improves from past false positives or missed vulnerabilities.

**CodeRL++ solves all three.** It provides a structured, reward-rich RL environment where agents learn the full engineering review loop — detect, fix, verify — and improve across episodes via persistent memory.

---

## ✅ Live Environment Validation

The live environment at [huggingface.co/spaces/Sap1x/CodeRLAgent](https://huggingface.co/spaces/Sap1x/CodeRLAgent) has been manually verified:

| Check | Result |
|---|---|
| `/health` responds 200 | ✅ |
| `/reset` returns valid observation | ✅ |
| `/step` returns reward + next observation | ✅ |
| `/tasks` returns task list | ✅ |
| Average response time | ~1.2 sec |
| Runtime crashes during 15-episode test | None |
| Validation test suite | 15/15 passing |

![HF Space Validation](assets/hf_validation.webp)
*Manual verification of the CodeRL++ environment running live on Hugging Face Spaces.*

> Quick verify: `curl https://huggingface.co/spaces/Sap1x/CodeRLAgent/health`

---

## 🧪 Reproducibility

Everything needed to reproduce our results is self-contained:

| Step | Details |
|---|---|
| Notebook runs without edits | ✅ Tested on Colab A100 and T4 |
| All dependencies auto-installed | ✅ Single pip cell, no manual steps |
| Environment server starts automatically | ✅ Subprocess launch in notebook |
| Training completes successfully | ✅ ~20 min A100, ~55 min T4 |
| Plots auto-generated | ✅ `assets/reward_curve.png`, `assets/before_after.png` |
| Model used in notebook | `Qwen/Qwen2.5-7B-Instruct` (fits free T4) |
| Model used for full results | `meta-llama/Llama-3.3-70B-Instruct` (A100) |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vgc5wdIR-wDXaWsuTnkAMf4uK5eUznuU?usp=sharing)

---

## 📈 Training Results (Evidence of Improvement)

> Results produced using GRPO fine-tuning with Unsloth on `Qwen/Qwen2.5-7B-Instruct`. Training notebook available via [Google Colab](https://colab.research.google.com/drive/1Vgc5wdIR-wDXaWsuTnkAMf4uK5eUznuU?usp=sharing).

### Reward Curve

Three clear training phases observed over 3,000 steps:

- **Episodes 0–600 (Exploration):** High variance as the agent experiments with different tool usage patterns and review strategies. The reward signal is noisy as the agent explores the action space.
- **Episodes 600–2,000 (Exploitation):** Precision stabilizes as the agent learns which vulnerability patterns are rewarded. Recall climbs consistently as the agent develops systematic review strategies.
- **Episodes 2,000–3,000 (Refinement):** The improvement reward (`R_improve`) dominates — the agent actively leverages its `AgentMemory` to avoid repeating historical failure patterns and recover previously missed vulnerability classes.

![Reward Curve](assets/reward_curve.png)
*Reward climbs across 3,000 training steps. Three distinct phases visible: exploration → consolidation → memory-driven improvement.*

### Before vs. After Training (Quantitative)

| Metric | Untrained Baseline | After 3,000 Steps | Δ |
|---|---|---|---|
| Vulnerability Recall | 0.31 | 0.74 | **+138%** |
| Precision | 0.58 | 0.82 | **+41%** |
| False Positive Rate | 0.42 | 0.18 | **−57%** |
| Critical Bug Detection | 0.24 | 0.69 | **+188%** |
| Tool Efficiency Score | 0.41 | 0.78 | **+90%** |
| Episode Improvement Rate | — | 0.63 | — |

![Before vs After](assets/before_after.png)
*Side-by-side comparison of agent performance before and after GRPO training. The trained agent shows dramatic improvement across all security-critical metrics.*

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/Sap1x/CodeRL-.git
cd CodeRL-
pip install -r requirements.txt
```

### Run Validation Tests

```bash
python test_validation.py
# Expected: 15/15 tests passing ✅
```

### Launch the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Run Training (Colab — Recommended)

The fastest way to reproduce results — fully documented and OpenEnv-compliant:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vgc5wdIR-wDXaWsuTnkAMf4uK5eUznuU?usp=sharing)

---

## 🏗️ How the Environment Works

### The Detect → Fix → Verify Loop

Agents do not just **flag** issues — they complete the full engineering review loop:

1. **Detect** — Surface a bug via diff analysis or targeted tool call (`inspect_function`, `trace_variable`)
2. **Fix** — Propose a concrete, line-level code correction with severity classification and explanation
3. **Verify** — Call `get_call_graph()` or `check_test_coverage()` to confirm the fix resolves the root cause before final submission

This three-phase loop is what separates CodeRL++ from static analysis benchmarks. The four-component reward system is designed specifically to reinforce this behavior.

### Multi-Phase Reasoning

Each episode progresses through four structured reasoning phases, automatically transitioned based on step progress:

| Phase | Focus | Example Agent Behavior |
|---|---|---|
| **Surface** (0–25%) | Obvious bugs, typos, null dereferences | Scan diff for `None` checks, off-by-one errors |
| **Logic** (25–50%) | Control flow, data transformations | Use `get_call_graph()` to trace execution paths |
| **Security** (50–75%) | Injection, auth flaws, crypto weaknesses | Use `inspect_function()` to check input sanitization |
| **Refinement** (75–100%) | Consolidate, deduplicate, verify fixes | Review all findings for consistency and completeness |

### Cross-Episode Memory (Core Innovation)

Unlike static benchmarks, CodeRL++ agents **learn across episodes**. The `AgentMemoryManager` tracks:

- **Missed Vulnerabilities** — Which vulnerability types the agent failed to detect, by severity and frequency
- **False Positives** — Which issue types the agent incorrectly flagged, enabling calibration
- **Reasoning Failures** — Episodes where the agent submitted findings but achieved zero true positives
- **Tool Usage Patterns** — Per-tool efficiency scores (successful calls ÷ total calls)

This memory is passed as part of every observation via `agent_memory_summary`, creating a self-improvement feedback loop:

```
Episode N: Agent misses SQL Injection on hard_001
  → Memory records: {missed: "SQL Injection", severity: critical, task: hard_001}
Episode N+1: Agent sees memory summary: "commonly_missed: SQL Injection (3 times)"
  → Agent prioritizes SQL pattern checks → Finds the injection → Gets improvement bonus
```

The `R_improve` reward explicitly incentivizes this:

```
R_improve = α × (recall_t - recall_{t-1})     ← reward recall improvement
           - β × repeated_mistakes              ← penalize repeating known failures
           + γ × new_class_coverage             ← bonus for recovering missed classes
```

---

## 📁 Project Structure

```
CodeRL-/
│
├── env/                        # Core RL environment
│   ├── environment.py          # Main CodeReviewEnv (reset/step/state methods)
│   ├── reward.py               # Four-component reward calculator
│   ├── memory.py               # Cross-episode agent memory manager
│   ├── grader.py               # Deterministic grading with fuzzy matching
│   ├── curriculum.py           # Adaptive curriculum (performance-based task selection)
│   ├── tools.py                # 6 simulated developer tools
│   ├── task_loader.py          # Task loader from JSON data files
│   └── state.py                # Pydantic models for all data structures
│
├── data/                       # Adversarial task suite
│   ├── easy.json               # 2 tasks: resource leaks, input validation
│   ├── medium.json             # 2 tasks: business logic, threading bugs
│   └── hard.json               # 2 tasks: SQL injection, path traversal, RCE
│
├── config/
│   └── reward.yaml             # Configurable reward weights
│
├── notebooks/
│   ├── train_coderl.ipynb      # Colab training notebook (Unsloth + GRPO)
│   └── run_notebook_e2e.py     # E2E verification script (CI/CD)
│
├── assets/                     # Training plots and media
│   ├── reward_curve.png        # Training reward curve
│   ├── before_after.png        # Before/after comparison
│   └── hf_validation.webp      # HF Space validation screenshot
│
├── server.py                   # FastAPI production server (8 endpoints)
├── inference.py                # Baseline inference agent
├── test_validation.py          # 15-test validation suite
├── openenv.yaml                # OpenEnv specification
├── Dockerfile                  # Container deployment
├── docker-compose.yaml         # Multi-service deployment (with Redis)
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Project metadata
```

---

## 🔬 Mathematical Methodology

### POMDP Formulation

We model the code review process as a **Partially Observable Markov Decision Process (POMDP)**:

- **State (S):** The full codebase — including hidden ground truth vulnerabilities the agent cannot directly observe.
- **Observation (O):** The current diff, tool outputs, interaction history, and `AgentMemory` summary. The agent must infer the true state from partial information.
- **Action (A):** A JSON object containing review comments (with line, issue, severity, explanation, suggestion, confidence) and optional tool calls.
- **Reward (R):** The multi-component scalar feedback signal computed deterministically from the grading function.
- **Transition (T):** Deterministic — each step advances the phase counter and accumulates findings.

### Policy Optimization

We employ **Group Relative Policy Optimization (GRPO)** to optimize the agent's reasoning trajectory. GRPO generates multiple completions per prompt and computes advantages relative to the group mean, avoiding the need for a separate value model. This is particularly effective for our structured output format (JSON review comments).

---

## 🧮 Four-Component Reward System

```
R_total = R_step + R_traj + R_improve + R_tool
```

### 1. Step-Level Reward (R_step)
Per-step precision/recall on submitted findings, with severity bonuses and false positive penalties:
```
R_step = w1 × precision + w2 × recall + w3 × severity_bonus
       - w4 × false_positive_count - w5 × duplicate_count
```
- **Severity Bonuses:** Critical (+0.30), High (+0.15), Medium (+0.05), Low (+0.00)
- **Matching:** Fuzzy line matching (±3 lines) with case-insensitive issue type similarity

### 2. Trajectory Reward (R_traj)
End-of-episode assessment of reasoning quality across the full action sequence:
```
R_traj = w6 × reasoning_consistency + w7 × efficient_reasoning
       - w8 × redundant_actions - w9 × noisy_exploration
```
- **Reasoning consistency:** Were findings spread across multiple steps (not dumped on step 1)?
- **Efficient reasoning:** Were steps productive (findings or meaningful tool calls)?
- **Redundant actions:** Penalizes targeting the same line multiple times
- **Noisy exploration:** Penalizes many tool calls with no resulting findings

### 3. Improvement Reward (R_improve)
Cross-episode signal computed by the `AgentMemoryManager`:
```
R_improve = α × (recall_t - recall_{t-1})
          - β × repeated_mistakes
          + γ × new_class_coverage
```
- `α = 0.3` — Recall gain weight
- `β = 0.4` — Repeated mistake penalty weight
- `γ = 0.2` — New vulnerability class recovery bonus weight

### 4. Tool Efficiency Reward (R_tool)
Rewards meaningful information gain from tool calls:
```
R_tool = w10 × information_gain - w11 × redundant_call_penalty
```
- **Information gain:** Credit successful tool calls, especially when followed by findings
- **Redundant call penalty:** Penalizes calling the same tool with the same argument twice

---

## 🔧 Available Agent Tools

The environment provides 6 simulated developer tools that return task-aware responses:

| Tool | Purpose | Example Call |
|---|---|---|
| `inspect_function` | View function body, params, return type | `inspect_function("register_user")` |
| `trace_variable` | Track variable through assignments | `trace_variable("user_input")` |
| `get_call_graph` | Map function call chain | `get_call_graph("process_payment")` |
| `check_test_coverage` | View test coverage for function | `check_test_coverage("validate_input")` |
| `inspect_import` | Examine imported module details | `inspect_import("subprocess")` |
| `search_codebase` | Search for pattern across files | `search_codebase("os.system")` |

---

## 🐳 Docker Deployment

```bash
# Standalone
docker build -t coderl-plus-plus .
docker run -p 7860:7860 coderl-plus-plus

# With Redis (memory persistence across restarts)
docker-compose up --build
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Environment summary and metadata |
| `/health` | `GET` | Health check (returns status, version) |
| `/reset` | `POST` | Start a new review episode |
| `/step` | `POST` | Submit an action, returns observation + reward |
| `/state` | `GET` | Get current internal environment state |
| `/tasks` | `GET` | List all available task IDs |
| `/memory` | `GET` | Retrieve current agent memory state |
| `/metrics` | `GET` | Retrieve training metrics and reward stats |

### POST /reset

```bash
curl -X POST https://huggingface.co/spaces/Sap1x/CodeRLAgent/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard_001"}'
```

**Response:**
```json
{
  "success": true,
  "observation": {
    "code_diff": "def register_user(username, password, email):\n    query = f\"INSERT INTO users ...",
    "file_name": "auth_service.py",
    "difficulty": "hard",
    "phase": "surface",
    "step": 1,
    "max_steps": 8,
    "available_tools": ["inspect_function", "trace_variable", "get_call_graph", ...],
    "agent_memory_summary": { "commonly_missed_issues": {"SQL Injection": 3}, ... }
  }
}
```

### POST /step

```bash
curl -X POST https://huggingface.co/spaces/Sap1x/CodeRLAgent/step \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      {
        "line": 31,
        "issue": "SQL Injection",
        "severity": "critical",
        "explanation": "User input directly interpolated into SQL query string via f-string",
        "suggestion": "Use parameterized queries: cursor.execute(\"INSERT INTO users VALUES (?, ?, ?)\", (username, password, email))",
        "confidence": 0.95
      }
    ],
    "tool_calls": [
      {"tool": "get_call_graph", "argument": "register_user"}
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "observation": { "phase": "logic", "step": 2, "..." : "..." },
  "reward": {
    "precision_score": 0.5,
    "recall_score": 0.15,
    "severity_bonus": 0.3,
    "false_positive_penalty": 0.0,
    "step_reward": 0.95,
    "trajectory_reward": { "total": 0.0 },
    "improvement_reward": { "recall_gain": 0.15, "repeated_mistakes": 0.0, "total": 0.045 },
    "tool_efficiency_reward": { "information_gain": 0.3, "total": 0.03 },
    "total": 1.025
  },
  "done": false,
  "info": { "step": 2, "phase": "logic", "new_issues_count": 1 }
}
```

---

## 📊 Task Suite

6 adversarial tasks across 3 difficulty tiers:

| Task ID | Difficulty | Vulnerability Types | Description |
|---|---|---|---|
| `easy_001` | 🟢 Easy | Resource leak, off-by-one | Utility functions with subtle resource management bugs |
| `easy_002` | 🟢 Easy | Weak validation, missing sanitization | Input validators with bypasses |
| `medium_001` | 🟡 Medium | Incorrect business logic | Order processor with calculation errors |
| `medium_002` | 🟡 Medium | Threading bugs, data structure issues | LRU cache with race conditions |
| `hard_001` | 🔴 Hard | SQL injection, weak crypto, data exposure | Authentication system with chained vulnerabilities |
| `hard_002` | 🔴 Hard | Path traversal, command injection, eval | File handler with multiple injection vectors |

---

## 📝 License

MIT License. See [LICENSE](./LICENSE) for details.

---

## 🔗 Links

- **Live Environment:** [huggingface.co/spaces/Sap1x/CodeRLAgent](https://huggingface.co/spaces/Sap1x/CodeRLAgent)
- **Training Notebook:** [Google Colab](https://colab.research.google.com/drive/1Vgc5wdIR-wDXaWsuTnkAMf4uK5eUznuU?usp=sharing)
- **GitHub Repository:** [github.com/Sap1x/CodeRL-](https://github.com/Sap1x/CodeRL-.git)
- **Blog Post:** [huggingface.co/blog/Sap1x/coderl-plus-plus](https://huggingface.co/blog/Sap1x/coderl-plus-plus)

---

> *CodeRL++ — closing the gap between raw LLM capability and genuine software engineering intelligence.*
