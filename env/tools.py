"""
CodeRL Tool Simulator — Simulated developer tools for multi-step reasoning.

Provides 6 tools that agents can invoke during code review:
    - inspect_function: Returns function signature and metadata
    - trace_variable: Returns variable usage chain
    - get_call_graph: Returns caller/callee relationships
    - check_test_coverage: Returns simulated coverage data
    - inspect_import: Examines module imports and usage
    - search_codebase: Searches for patterns across code
"""

from __future__ import annotations

import re
from typing import Any, Optional

from env.state import Task


# ──────────────────────────────────────────────
# Tool Simulator
# ──────────────────────────────────────────────

class ToolSimulator:
    """
    Simulates developer tools that an agent can invoke during code review.

    Supported tools:
        - inspect_function: Returns function signature and docstring
        - trace_variable: Returns variable usage chain
        - get_call_graph: Returns caller/callee relationships for a function
        - check_test_coverage: Returns simulated test coverage data
        - inspect_import: Examines what is imported and how it's used
        - search_codebase: Searches entire repo for a pattern or symbol
    """

    def __init__(self, task: Task):
        self._task = task
        self._call_log: list[dict[str, Any]] = []

    # ── public API ──────────────────────────────

    def execute(self, tool_name: str, argument: str) -> dict[str, Any]:
        """
        Execute a simulated tool.

        Args:
            tool_name: One of the supported tool names
            argument: The function/variable/pattern to inspect

        Returns:
            Tool result dict with 'success', 'tool', 'argument', and 'result' keys
        """
        handlers = {
            "inspect_function": self._inspect_function,
            "trace_variable": self._trace_variable,
            "get_call_graph": self._get_call_graph,
            "check_test_coverage": self._check_test_coverage,
            "inspect_import": self._inspect_import,
            "search_codebase": self._search_codebase,
        }

        handler = handlers.get(tool_name)
        if not handler:
            result = {
                "success": False,
                "tool": tool_name,
                "argument": argument,
                "error": f"Unknown tool: '{tool_name}'. Available: {list(handlers.keys())}",
            }
        else:
            result = handler(argument)

        # Log the call
        self._call_log.append({
            "tool": tool_name,
            "argument": argument,
            "success": result.get("success", False),
        })

        return result

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Return the history of tool calls."""
        return list(self._call_log)

    # ── tool implementations ────────────────────

    def _inspect_function(self, function_name: str) -> dict[str, Any]:
        """Return function signature and metadata."""
        # Check task-level function signatures first
        if self._task.function_signatures and function_name in self._task.function_signatures:
            signature = self._task.function_signatures[function_name]
            return {
                "success": True,
                "tool": "inspect_function",
                "argument": function_name,
                "result": {
                    "name": function_name,
                    "signature": signature,
                    "found_in": self._task.file_name,
                },
            }

        # Try to extract from the code diff
        extracted = self._extract_function_from_diff(function_name)
        if extracted:
            return {
                "success": True,
                "tool": "inspect_function",
                "argument": function_name,
                "result": extracted,
            }

        return {
            "success": False,
            "tool": "inspect_function",
            "argument": function_name,
            "error": f"Function '{function_name}' not found in the current context.",
        }

    def _trace_variable(self, variable_name: str) -> dict[str, Any]:
        """Return variable usage trace."""
        # Check task-level variable traces
        if self._task.variable_traces and variable_name in self._task.variable_traces:
            trace = self._task.variable_traces[variable_name]
            return {
                "success": True,
                "tool": "trace_variable",
                "argument": variable_name,
                "result": {
                    "variable": variable_name,
                    "usage_locations": trace,
                    "file": self._task.file_name,
                },
            }

        # Try to find in code diff
        locations = self._find_variable_in_diff(variable_name)
        if locations:
            return {
                "success": True,
                "tool": "trace_variable",
                "argument": variable_name,
                "result": {
                    "variable": variable_name,
                    "usage_locations": locations,
                    "file": self._task.file_name,
                },
            }

        return {
            "success": False,
            "tool": "trace_variable",
            "argument": variable_name,
            "error": f"Variable '{variable_name}' not found in the current context.",
        }

    def _get_call_graph(self, function_name: str) -> dict[str, Any]:
        """
        Return caller/callee relationships for a function.

        Parses the code diff to find:
        - Which functions call the target function (callers)
        - Which functions the target function calls (callees)
        """
        lines = self._task.code_diff.split("\n")

        # Find the target function body
        callers: list[str] = []
        callees: list[str] = []
        in_function = False
        current_func: Optional[str] = None

        for line in lines:
            stripped = line.lstrip("+").lstrip("-").lstrip()

            # Track which function we're in
            if stripped.startswith("def "):
                match = re.match(r"def\s+(\w+)\s*\(", stripped)
                if match:
                    current_func = match.group(1)
                    in_function = current_func == function_name

            # If we're inside the target function, find callees
            if in_function and current_func == function_name:
                calls = re.findall(r"(\w+)\s*\(", stripped)
                for call in calls:
                    if call != function_name and call not in ("def", "if", "for", "while", "print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple", "isinstance", "type", "return"):
                        if call not in callees:
                            callees.append(call)

            # If we're in any other function, check if it calls our target
            if current_func and current_func != function_name:
                if f"{function_name}(" in stripped:
                    if current_func not in callers:
                        callers.append(current_func)

        # Also check related files
        if self._task.related_files:
            for fname, content in self._task.related_files.items():
                if f"{function_name}(" in content:
                    callers.append(f"(from {fname})")

        if callers or callees:
            return {
                "success": True,
                "tool": "get_call_graph",
                "argument": function_name,
                "result": {
                    "function": function_name,
                    "callers": callers,
                    "callees": callees,
                    "file": self._task.file_name,
                },
            }

        return {
            "success": False,
            "tool": "get_call_graph",
            "argument": function_name,
            "error": f"Function '{function_name}' not found in call graph.",
        }

    def _check_test_coverage(self, file_name: str) -> dict[str, Any]:
        """
        Return simulated test coverage data for a file.

        Generates coverage based on the code diff — lines with known
        ground truth issues are marked as uncovered (simulating the
        gap that tests don't catch the bugs).
        """
        lines = self._task.code_diff.split("\n")
        total_lines = len(lines)

        # Ground truth issue lines are "uncovered"
        uncovered_lines = [gt.line for gt in self._task.ground_truth]

        # Added lines (+) that aren't in ground truth are "covered"
        covered_lines = []
        for i, line in enumerate(lines, 1):
            if line.startswith("+") and not line.startswith("+++"):
                if i not in uncovered_lines:
                    covered_lines.append(i)

        coverage_pct = (
            len(covered_lines) / (len(covered_lines) + len(uncovered_lines)) * 100
            if (covered_lines or uncovered_lines)
            else 0.0
        )

        return {
            "success": True,
            "tool": "check_test_coverage",
            "argument": file_name,
            "result": {
                "file": file_name,
                "coverage_percentage": round(coverage_pct, 1),
                "total_lines": total_lines,
                "covered_lines": len(covered_lines),
                "uncovered_lines": uncovered_lines[:15],  # Cap output
                "note": "Lines with low coverage may contain untested edge cases or bugs.",
            },
        }

    def _inspect_import(self, module_name: str) -> dict[str, Any]:
        """
        Examine what is imported from a module and how it's used.

        Parses import statements and usage patterns from the code diff.
        """
        lines = self._task.code_diff.split("\n")
        imports_found: list[str] = []
        usage_locations: list[str] = []

        for i, line in enumerate(lines, 1):
            stripped = line.lstrip("+").lstrip("-").lstrip()

            # Check import statements
            if f"import {module_name}" in stripped or f"from {module_name}" in stripped:
                prefix = "added" if line.startswith("+") else "removed" if line.startswith("-") else "context"
                imports_found.append(f"line ~{i} ({prefix}): {stripped.strip()}")

            # Check usage (not import lines)
            elif module_name in stripped and "import" not in stripped:
                prefix = "added" if line.startswith("+") else "removed" if line.startswith("-") else "context"
                usage_locations.append(f"line ~{i} ({prefix}): {stripped.strip()}")

        # Also check related files
        related_usage: dict[str, list[str]] = {}
        if self._task.related_files:
            for fname, content in self._task.related_files.items():
                if module_name in content:
                    rlines = [
                        l.strip() for l in content.split("\n")
                        if module_name in l
                    ]
                    if rlines:
                        related_usage[fname] = rlines[:5]

        if imports_found or usage_locations:
            return {
                "success": True,
                "tool": "inspect_import",
                "argument": module_name,
                "result": {
                    "module": module_name,
                    "imports": imports_found,
                    "usage_in_diff": usage_locations[:10],
                    "usage_in_related_files": related_usage,
                    "file": self._task.file_name,
                },
            }

        return {
            "success": False,
            "tool": "inspect_import",
            "argument": module_name,
            "error": f"Module '{module_name}' not found in the current context.",
        }

    def _search_codebase(self, pattern: str) -> dict[str, Any]:
        """
        Search entire repo for a pattern or symbol.

        Searches through code_diff and related_files.
        """
        results: list[dict[str, Any]] = []

        # Search in main diff
        lines = self._task.code_diff.split("\n")
        for i, line in enumerate(lines, 1):
            content = line.lstrip("+").lstrip("-").lstrip()
            if pattern.lower() in content.lower():
                prefix = "added" if line.startswith("+") else "removed" if line.startswith("-") else "context"
                results.append({
                    "file": self._task.file_name,
                    "line": i,
                    "type": prefix,
                    "content": content.strip(),
                })

        # Search in related files
        if self._task.related_files:
            for fname, content in self._task.related_files.items():
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern.lower() in line.lower():
                        results.append({
                            "file": fname,
                            "line": i,
                            "type": "related",
                            "content": line.strip(),
                        })

        if results:
            return {
                "success": True,
                "tool": "search_codebase",
                "argument": pattern,
                "result": {
                    "pattern": pattern,
                    "matches": results[:20],  # Cap at 20 results
                    "total_matches": len(results),
                },
            }

        return {
            "success": False,
            "tool": "search_codebase",
            "argument": pattern,
            "error": f"Pattern '{pattern}' not found in the codebase.",
        }

    # ── helpers ─────────────────────────────────

    def _extract_function_from_diff(self, function_name: str) -> Optional[dict]:
        """Try to extract function info from the code diff text."""
        lines = self._task.code_diff.split("\n")
        for i, line in enumerate(lines):
            stripped = line.lstrip("+").lstrip("-").lstrip()
            if stripped.startswith(f"def {function_name}("):
                return {
                    "name": function_name,
                    "signature": stripped.rstrip(":").strip(),
                    "found_in": self._task.file_name,
                    "approximate_line": i + 1,
                }
        return None

    def _find_variable_in_diff(self, variable_name: str) -> list[str]:
        """Find all lines in the diff that reference a variable."""
        locations = []
        lines = self._task.code_diff.split("\n")
        for i, line in enumerate(lines):
            content = line.lstrip("+").lstrip("-").lstrip()
            if variable_name in content:
                prefix = "added" if line.startswith("+") else "removed" if line.startswith("-") else "context"
                locations.append(f"line ~{i + 1} ({prefix}): {content.strip()}")
        return locations[:10]  # Cap at 10 results
