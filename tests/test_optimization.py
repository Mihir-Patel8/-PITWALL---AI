import pytest

from pitwall_ai.strategy import optimizer
from pitwall_ai.strategy.estimators import ParametricEstimator
from pitwall_ai.strategy.simulator import SimulationConfig, simulate_strategy

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def make_estimator(deg: float) -> ParametricEstimator:
    return ParametricEstimator(
        90.0,
        dict.fromkeys(COMPOUNDS, 0.0),
        dict.fromkeys(COMPOUNDS, deg),
        fuel_gain_per_lap=0.0,
    )


def test_candidates_satisfy_all_constraints():
    cands = optimizer.generate_candidates(
        30, "MEDIUM", COMPOUNDS, min_stops=1, max_stops=2, min_stint=6, max_stint=20, step=2
    )
    assert cands
    for strategy in cands:
        assert strategy.total_laps == 30
        assert 1 <= strategy.n_stops <= 2
        assert strategy.stints[0].compound == "MEDIUM"
        assert len({s.compound for s in strategy.stints}) >= 2
        assert all(6 <= s.laps <= 20 for s in strategy.stints)
        strategy.validate(30, allowed_compounds=COMPOUNDS, min_stint=6, require_two_compounds=True)
    assert len({s.label() for s in cands}) == len(cands)  # unique names


def test_step_reduces_search_space():
    fine = optimizer.generate_candidates(40, "SOFT", COMPOUNDS, max_stops=1, min_stint=5, step=1)
    coarse = optimizer.generate_candidates(40, "SOFT", COMPOUNDS, max_stops=1, min_stint=5, step=5)
    assert 0 < len(coarse) < len(fine)


def test_zero_stop_allowed_only_without_two_compound_rule():
    cands = optimizer.generate_candidates(
        20, "HARD", COMPOUNDS, min_stops=0, max_stops=0, require_two_compounds=False
    )
    assert len(cands) == 1 and cands[0].n_stops == 0
    with pytest.raises(ValueError, match="No valid strategies"):
        optimizer.generate_candidates(20, "HARD", COMPOUNDS, min_stops=0, max_stops=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_laps": 0},
        {"start_compound": "WET"},
        {"min_stops": 3, "max_stops": 1},
        {"min_stint": 0},
        {"step": 0},
        {"min_stint": 10, "max_stint": 5},
        {"min_stint": 30},  # cannot fit even one stop in 40 laps
    ],
)
def test_invalid_parameter_combinations(kwargs):
    params = {"total_laps": 40, "start_compound": "SOFT", "available_compounds": COMPOUNDS} | kwargs
    with pytest.raises(ValueError):
        optimizer.generate_candidates(**params)


def test_candidate_cap_protects_against_huge_searches():
    with pytest.raises(ValueError, match="candidates"):
        optimizer.generate_candidates(60, "SOFT", COMPOUNDS, max_stops=2, max_candidates=10)


def test_best_candidate_has_lowest_calculated_time():
    est = make_estimator(0.05)
    cfg = SimulationConfig(40, 20.0)
    cands = optimizer.generate_candidates(40, "MEDIUM", COMPOUNDS, max_stops=2, min_stint=8)
    ranking = optimizer.rank_strategies(cands, est, cfg)
    independent = min(simulate_strategy(s, est, cfg).total_time for s in cands)
    assert ranking["TotalTime"].iloc[0] == pytest.approx(independent)
    assert ranking["TotalTime"].is_monotonic_increasing
    assert ranking["DeltaToBest"].iloc[0] == 0 and (ranking["DeltaToBest"] >= 0).all()
    assert ranking["Rank"].tolist() == list(range(1, len(ranking) + 1))


def test_ranking_reflects_physics_one_stop_vs_two_stop():
    cfg = SimulationConfig(40, 20.0)
    # No tyre wear: every extra stop only costs time -> best plan is a one-stop.
    flat = optimizer.rank_strategies(
        optimizer.generate_candidates(40, "MEDIUM", COMPOUNDS, max_stops=2, min_stint=8),
        make_estimator(0.0),
        cfg,
    )
    assert flat.iloc[0]["Stops"] == 1
    # Brutal wear, cheap stops -> two stops win.
    harsh = optimizer.rank_strategies(
        optimizer.generate_candidates(40, "MEDIUM", COMPOUNDS, max_stops=2, min_stint=8),
        make_estimator(0.5),
        SimulationConfig(40, 5.0),
    )
    assert harsh.iloc[0]["Stops"] == 2
    by_stops = optimizer.best_by_stop_count(harsh)
    assert by_stops["Stops"].tolist() == [1, 2]


def test_results_are_reproducible():
    est = ParametricEstimator.demo_defaults()
    cfg = SimulationConfig(30)
    run = lambda: optimizer.rank_strategies(  # noqa: E731
        optimizer.generate_candidates(30, "MEDIUM", COMPOUNDS, min_stint=8), est, cfg
    )
    assert run().equals(run())


def test_explanation_mentions_best_strategy_and_caveat():
    est = make_estimator(0.05)
    ranking = optimizer.rank_strategies(
        optimizer.generate_candidates(30, "MEDIUM", COMPOUNDS, min_stint=8),
        est,
        SimulationConfig(30),
    )
    text = optimizer.explain_choice(ranking)
    assert ranking.iloc[0]["Strategy"] in text
    assert "not necessarily the optimal real-world decision" in text
    with pytest.raises(ValueError):
        optimizer.rank_strategies([], est, SimulationConfig(30))
