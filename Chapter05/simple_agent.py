"""Simple agent demonstrating planning + tool use inspired by the book.

This script avoids external LLM dependencies and focuses on the core
agentic loop: plan -> act with tools -> reflect in memory.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from typing import Callable, Iterable, List


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[str], str]

    def run(self, input_text: str) -> str:
        return self.handler(input_text)


class CalculatorTool:
    """Safely evaluates arithmetic expressions."""

    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.UnaryOp,
        ast.Constant,
    }

    @staticmethod
    def _validate(node: ast.AST) -> None:
        if type(node) not in CalculatorTool.allowed_nodes:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            CalculatorTool._validate(child)

    def __call__(self, expression: str) -> str:
        if not expression.strip():
            return "No expression provided."
        tree = ast.parse(expression, mode="eval")
        self._validate(tree)
        result = eval(compile(tree, filename="<calculator>", mode="eval"))
        return f"{expression.strip()} = {result}"


class EchoTool:
    def __call__(self, message: str) -> str:
        return message.strip() or "(no message)"


class ChecklistTool:
    def __call__(self, text: str) -> str:
        items = [item.strip() for item in text.split(",") if item.strip()]
        if not items:
            return "No checklist items found."
        return "\n".join(f"- [ ] {item}" for item in items)


@dataclass
class AgentMemory:
    events: List[str] = field(default_factory=list)

    def add(self, entry: str) -> None:
        self.events.append(entry)

    def summary(self) -> str:
        if not self.events:
            return "Memory is empty."
        return "\n".join(f"* {event}" for event in self.events)


class SimpleAgent:
    def __init__(self, tools: Iterable[Tool]):
        self.tools = {tool.name: tool for tool in tools}
        self.memory = AgentMemory()

    def plan(self, goal: str) -> List[str]:
        raw_steps = [step.strip() for step in goal.split(" and ") if step.strip()]
        if not raw_steps:
            return [goal.strip()]
        return raw_steps

    def route_tool(self, step: str) -> str:
        lowered = step.lower()
        if any(keyword in lowered for keyword in ["calculate", "compute", "sum", "total"]):
            return "calculator"
        if any(keyword in lowered for keyword in ["checklist", "list", "todos"]):
            return "checklist"
        return "echo"

    def act(self, step: str) -> str:
        tool_name = self.route_tool(step)
        tool = self.tools[tool_name]
        result = tool.run(step)
        self.memory.add(f"Tool {tool.name} handled '{step}' -> {result}")
        return result

    def run(self, goal: str) -> List[str]:
        self.memory.add(f"Goal received: {goal}")
        plan = self.plan(goal)
        self.memory.add(f"Plan: {plan}")
        outputs = [self.act(step) for step in plan]
        self.memory.add("Run complete.")
        return outputs


def build_agent() -> SimpleAgent:
    return SimpleAgent(
        tools=[
            Tool("calculator", "Evaluate arithmetic expressions", CalculatorTool()),
            Tool("echo", "Repeat or summarize text", EchoTool()),
            Tool("checklist", "Turn comma-separated items into a checklist", ChecklistTool()),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple agentic loop.")
    parser.add_argument("goal", help="Describe the goal for the agent to execute.")
    args = parser.parse_args()

    agent = build_agent()
    results = agent.run(args.goal)

    print("Agent outputs:\n")
    for result in results:
        print(result)
    print("\nAgent memory:\n")
    print(agent.memory.summary())


if __name__ == "__main__":
    main()
