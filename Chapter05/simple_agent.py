"""MVP causal agent based on the chapter pipeline diagram.

Pipeline implemented:
1) Background knowledge -> assumptions
2) Build causal model
3) Check if query can be answered
4) If yes, derive estimand and perform statistical estimation
5) Return estimate + testable implications
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence


@dataclass
class CausalInputs:
    knowledge: str
    assumptions: List[str]
    query: str
    treatment_col: str
    outcome_col: str
    data_path: str
    confounders: List[str] = field(default_factory=list)


@dataclass
class CausalOutputs:
    testable_implications: List[str]
    query_answerable: bool
    estimand: str | None
    estimate: float | None
    units_treated: int
    units_control: int
    notes: List[str]


class CausalMVPEngine:
    """Small inference engine with a practical fallback loop."""

    def __init__(self, inputs: CausalInputs):
        self.inputs = inputs

    def build_causal_model(self) -> Dict[str, object]:
        return {
            "nodes": [self.inputs.treatment_col, self.inputs.outcome_col, *self.inputs.confounders],
            "edges": [
                [self.inputs.treatment_col, self.inputs.outcome_col],
                *[[c, self.inputs.treatment_col] for c in self.inputs.confounders],
                *[[c, self.inputs.outcome_col] for c in self.inputs.confounders],
            ],
            "assumptions": self.inputs.assumptions,
        }

    def testable_implications(self) -> List[str]:
        return [
            f"{self.inputs.treatment_col} and {self.inputs.outcome_col} columns exist.",
            f"{self.inputs.treatment_col} is binary (0/1).",
            "Both treatment and control groups have at least one row.",
            "Outcome column is numeric.",
        ]

    def can_answer_query(self, rows: Sequence[dict]) -> tuple[bool, List[str]]:
        notes: List[str] = []
        if not rows:
            notes.append("Dataset is empty.")
            return False, notes

        required = {self.inputs.treatment_col, self.inputs.outcome_col}
        missing = [c for c in required if c not in rows[0]]
        if missing:
            notes.append(f"Missing required columns: {missing}.")
            return False, notes

        treated = [r for r in rows if str(r[self.inputs.treatment_col]).strip() == "1"]
        control = [r for r in rows if str(r[self.inputs.treatment_col]).strip() == "0"]
        if not treated or not control:
            notes.append("Need both treatment=1 and treatment=0 rows.")
            return False, notes

        try:
            _ = [float(r[self.inputs.outcome_col]) for r in rows]
        except ValueError:
            notes.append("Outcome column must be numeric.")
            return False, notes

        return True, notes

    def derive_estimand(self) -> str:
        return f"E[{self.inputs.outcome_col}|do({self.inputs.treatment_col}=1)] - E[{self.inputs.outcome_col}|do({self.inputs.treatment_col}=0)]"

    def estimate_effect(self, rows: Sequence[dict]) -> tuple[float, int, int]:
        treated_outcomes = [
            float(r[self.inputs.outcome_col])
            for r in rows
            if str(r[self.inputs.treatment_col]).strip() == "1"
        ]
        control_outcomes = [
            float(r[self.inputs.outcome_col])
            for r in rows
            if str(r[self.inputs.treatment_col]).strip() == "0"
        ]
        ate = mean(treated_outcomes) - mean(control_outcomes)
        return ate, len(treated_outcomes), len(control_outcomes)

    def run(self) -> CausalOutputs:
        rows = load_csv(self.inputs.data_path)
        answerable, notes = self.can_answer_query(rows)

        if not answerable:
            notes.append("Return to assumptions/model: refine DAG, data, or query scope.")
            return CausalOutputs(
                testable_implications=self.testable_implications(),
                query_answerable=False,
                estimand=None,
                estimate=None,
                units_treated=0,
                units_control=0,
                notes=notes,
            )

        estimand = self.derive_estimand()
        estimate, n_treated, n_control = self.estimate_effect(rows)
        notes.append("MVP estimator uses difference-in-means; upgrade to DML/DR for production.")

        return CausalOutputs(
            testable_implications=self.testable_implications(),
            query_answerable=True,
            estimand=estimand,
            estimate=round(estimate, 6),
            units_treated=n_treated,
            units_control=n_control,
            notes=notes,
        )


def load_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_demo_data(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"user_id": 1, "treatment": 1, "outcome": 1.0, "recency": 2},
        {"user_id": 2, "treatment": 1, "outcome": 0.8, "recency": 5},
        {"user_id": 3, "treatment": 1, "outcome": 1.2, "recency": 3},
        {"user_id": 4, "treatment": 0, "outcome": 0.5, "recency": 4},
        {"user_id": 5, "treatment": 0, "outcome": 0.6, "recency": 8},
        {"user_id": 6, "treatment": 0, "outcome": 0.4, "recency": 6},
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "treatment", "outcome", "recency"])
        writer.writeheader()
        writer.writerows(rows)


def demo_inputs(demo_csv: str) -> CausalInputs:
    return CausalInputs(
        knowledge="Targeted messaging can influence subscription conversion.",
        assumptions=[
            "No unobserved confounding after recency adjustment.",
            "Stable treatment definition.",
            "Positivity holds in observed data.",
        ],
        query="What is the causal effect of treatment on conversion?",
        treatment_col="treatment",
        outcome_col="outcome",
        data_path=demo_csv,
        confounders=["recency"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP causal inference agent.")
    parser.add_argument("--input-json", help="Optional JSON file matching CausalInputs fields.")
    parser.add_argument("--create-demo-data", action="store_true", help="Write a demo CSV and run with demo settings.")
    parser.add_argument("--demo-csv", default="Chapter05/data/mvp_demo.csv", help="Path for demo CSV data.")
    args = parser.parse_args()

    if args.create_demo_data:
        write_demo_data(Path(args.demo_csv))

    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        inputs = CausalInputs(**payload)
    else:
        inputs = demo_inputs(args.demo_csv)

    model = CausalMVPEngine(inputs)
    outputs = model.run()

    result = {
        "inputs": asdict(inputs),
        "model": model.build_causal_model(),
        "outputs": asdict(outputs),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
