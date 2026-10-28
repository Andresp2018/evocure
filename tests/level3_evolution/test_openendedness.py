"""Activity statistics must be able to say "this system settled".

The value of the measurement is that it can return an unflattering answer. A
tracker that always reports open-endedness would be decoration.
"""

from __future__ import annotations

import pytest

from simulation.evolution.openendedness import (
    EvolutionaryActivityTracker,
    shannon_diversity,
)


def test_a_static_population_is_class_one():
    """No new components ever adopted, so it cannot be open ended."""
    tracker = EvolutionaryActivityTracker(novelty_window=5)
    for step in range(60):
        tracker.observe(step, ["A", "B", "C"])

    result = tracker.classify()
    assert result["class"] == "class 1"
    assert result["mean_new_activity_second_half"] == 0.0
    assert "not open ended" in result["reading"]


def test_a_system_that_innovates_then_settles_is_class_two():
    tracker = EvolutionaryActivityTracker(novelty_window=10)
    for step in range(30):
        tracker.observe(step, [f"C{i}" for i in range(3 + step // 10)])
    # Growth stops; the same components churn without diversity rising.
    for step in range(30, 80):
        tracker.observe(step, [f"C{i}" for i in range(6)] + [f"T{step}"])

    result = tracker.classify()
    assert result["class"] == "class 2"
    assert result["mean_new_activity_second_half"] > 0.0


def test_steadily_growing_diversity_reads_as_suggestive_class_three():
    tracker = EvolutionaryActivityTracker(novelty_window=20)
    for step in range(80):
        tracker.observe(step, [f"C{i}" for i in range(3 + step)])

    result = tracker.classify()
    assert result["class"].startswith("class 3")
    assert result["diversity_trend_per_step"] > 0
    assert "suggestive" in result["class"]


def test_class_three_never_claims_more_than_it_can_show():
    """Without a neutral shadow model, adaptation cannot be separated from drift."""
    tracker = EvolutionaryActivityTracker(novelty_window=20)
    for step in range(80):
        tracker.observe(step, [f"C{i}" for i in range(3 + step)])

    result = tracker.classify()
    assert "shadow model" in result["reading"] or "shadow" in result["limitation"]
    assert "not separated from neutral drift" in result["limitation"]


def test_too_few_observations_is_undetermined_not_a_verdict():
    tracker = EvolutionaryActivityTracker()
    for step in range(3):
        tracker.observe(step, ["A"])

    assert tracker.classify()["class"] == "undetermined"


def test_activity_accrues_only_while_a_component_persists():
    tracker = EvolutionaryActivityTracker(novelty_window=2)
    for step in range(10):
        tracker.observe(step, ["persistent"])
    tracker.observe(10, ["persistent", "latecomer"])

    assert tracker.records["persistent"].activity == 11.0
    assert tracker.records["latecomer"].activity == 1.0
    assert tracker.records["latecomer"].first_seen_step == 10


def test_new_activity_decays_as_a_component_stops_being_new():
    tracker = EvolutionaryActivityTracker(novelty_window=3)
    for step in range(10):
        snapshot = tracker.observe(step, ["A"])

    # Past the novelty window, A no longer counts towards new activity.
    assert snapshot.new_activity == 0.0
    assert snapshot.diversity == 1


def test_snapshot_records_population_and_diversity_separately():
    tracker = EvolutionaryActivityTracker()
    snapshot = tracker.observe(0, ["A", "A", "A", "B"])

    assert snapshot.population == 4
    assert snapshot.diversity == 2


def test_shannon_diversity_behaves():
    assert shannon_diversity([]) == 0.0
    assert shannon_diversity(["A"] * 10) == pytest.approx(0.0)
    even = shannon_diversity(["A", "B", "C", "D"])
    skewed = shannon_diversity(["A"] * 7 + ["B", "C", "D"])
    assert even > skewed


def test_summary_includes_a_sampled_series():
    tracker = EvolutionaryActivityTracker()
    for step in range(100):
        tracker.observe(step, [f"C{step % 5}"])

    summary = tracker.summary()
    assert summary["series"]
    assert len(summary["series"]) <= 45
    assert all({"step", "diversity", "new_activity"} <= set(point) for point in summary["series"])
