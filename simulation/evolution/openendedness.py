"""Measure whether the ecosystem keeps producing novelty, or settles.

Open ended evolution is the standing grand challenge in artificial life: a system
that converges to one optimum within a few hundred steps is a hill climber with
extra steps. The usual instrument is Bedau and Packard's evolutionary activity
statistics, which ask not "is there variation" but "is new variation still being
adopted, indefinitely".

The three classes those statistics distinguish:

  Class 1   no adaptive activity; drift only
  Class 2   bounded adaptive activity; the system innovates and then settles
  Class 3   unbounded adaptive activity; novelty keeps being adopted

What is implemented here. Per component activity counters, diversity, cumulative
and new activity, and trend tests over the run, which separate a system that has
clearly settled from one that is still moving.

What is not implemented, stated plainly because the distinction matters. The full
Bedau test compares against a neutral shadow model, a matched population with
inheritance but no selection, and calls activity adaptive only where it exceeds
what neutral drift produces. Without that shadow this cannot prove adaptation,
only describe the activity. A class 3 reading here is suggestive and not a claim
of open endedness; a class 1 reading is solid, because a system producing no new
activity at all cannot be open ended under any shadow model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class ActivityRecord:
    """One component's history: when it appeared and how long it has persisted."""

    first_seen_step: int
    last_seen_step: int
    activity: float = 0.0
    peak_abundance: int = 0


@dataclass
class ActivitySnapshot:
    step: int
    diversity: int
    population: int
    total_activity: float
    mean_activity: float
    new_activity: float
    new_components: int


class EvolutionaryActivityTracker:
    """Bedau-Packard style activity statistics over a run.

    A component is whatever identity you choose to track. Passing compound ids
    measures chemical novelty; passing lineage ids measures genealogical novelty.
    Chemical novelty is the one that matters for a chemical evolution claim.
    """

    def __init__(self, novelty_window: int = 50) -> None:
        # A component counts as new while it is within this many steps of first
        # being seen, which is what makes "new activity" distinct from "activity".
        self.novelty_window = int(novelty_window)
        self.records: dict[str, ActivityRecord] = {}
        self.snapshots: list[ActivitySnapshot] = []

    def observe(self, step: int, component_ids: Iterable[str]) -> ActivitySnapshot:
        """Record one time step. Each present component accrues one unit of activity."""
        counts: dict[str, int] = defaultdict(int)
        for component in component_ids:
            counts[str(component)] += 1

        new_activity, new_components = 0.0, 0
        for component, abundance in counts.items():
            record = self.records.get(component)
            if record is None:
                record = ActivityRecord(first_seen_step=step, last_seen_step=step)
                self.records[component] = record
                new_components += 1
            record.last_seen_step = step
            record.activity += 1.0
            record.peak_abundance = max(record.peak_abundance, abundance)
            if step - record.first_seen_step <= self.novelty_window:
                new_activity += 1.0

        present = [self.records[c].activity for c in counts]
        snapshot = ActivitySnapshot(
            step=step,
            diversity=len(counts),
            population=sum(counts.values()),
            total_activity=float(sum(present)),
            mean_activity=float(np.mean(present)) if present else 0.0,
            new_activity=new_activity,
            new_components=new_components,
        )
        self.snapshots.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------ reading

    def _trend(self, values: list[float]) -> float:
        """Slope per step over the second half, where transients have passed."""
        if len(values) < 6:
            return 0.0
        tail = values[len(values) // 2:]
        x = np.arange(len(tail), dtype=float)
        y = np.asarray(tail, dtype=float)
        if np.allclose(y, y[0]):
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    def classify(self) -> dict[str, Any]:
        if len(self.snapshots) < 6:
            return {
                "class": "undetermined",
                "reason": "too few observations to judge a trend",
                "steps_observed": len(self.snapshots),
            }

        diversity = [s.diversity for s in self.snapshots]
        new_activity = [s.new_activity for s in self.snapshots]
        mean_activity = [s.mean_activity for s in self.snapshots]

        diversity_trend = self._trend([float(d) for d in diversity])
        new_trend = self._trend(new_activity)
        tail_new = float(np.mean(new_activity[len(new_activity) // 2:]))

        if tail_new <= 0.0:
            label, reading = "class 1", (
                "No new components are being adopted in the second half of the run. "
                "Whatever variation exists is not producing anything that persists, "
                "so this is not open ended under any shadow model.")
        elif diversity_trend > 0.01 and new_trend >= -0.01:
            label, reading = "class 3 (suggestive)", (
                "Diversity is still rising and new components keep being adopted. "
                "This is consistent with unbounded activity, but without a neutral "
                "shadow model it cannot be distinguished from drift generating "
                "novelty that selection never acts on.")
        else:
            label, reading = "class 2", (
                "New components are adopted but diversity has stopped growing. The "
                "system innovated and then settled, which is bounded activity.")

        return {
            "class": label,
            "reading": reading,
            "steps_observed": len(self.snapshots),
            "components_ever_seen": len(self.records),
            "diversity_first": diversity[0],
            "diversity_last": diversity[-1],
            "diversity_trend_per_step": round(diversity_trend, 5),
            "new_activity_trend_per_step": round(new_trend, 5),
            "mean_new_activity_second_half": round(tail_new, 3),
            "mean_cumulative_activity": round(float(np.mean(mean_activity)), 3),
            "novelty_window_steps": self.novelty_window,
            "limitation": (
                "No neutral shadow model is run, so adaptive activity is not separated "
                "from neutral drift. A class 1 result is firm; a class 3 result is "
                "suggestive only."
            ),
        }

    def summary(self) -> dict[str, Any]:
        payload = self.classify()
        # Ceiling division, so a long run never exceeds roughly 40 sampled points.
        stride = max(1, -(-len(self.snapshots) // 40))
        payload["series"] = [
            {"step": s.step, "diversity": s.diversity, "population": s.population,
             "new_activity": s.new_activity, "new_components": s.new_components,
             "mean_activity": round(s.mean_activity, 3)}
            for s in self.snapshots[::stride]
        ]
        return payload


def shannon_diversity(component_ids: Iterable[str]) -> float:
    """Shannon entropy of the component distribution, in nats."""
    counts: dict[str, int] = defaultdict(int)
    for component in component_ids:
        counts[str(component)] += 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    proportions = np.array([c / total for c in counts.values()], dtype=float)
    nonzero = proportions[proportions > 0]
    return float(-(nonzero * np.log(nonzero)).sum())
