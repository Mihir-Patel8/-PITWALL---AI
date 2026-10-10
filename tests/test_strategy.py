import numpy as np
import pytest

from pitwall_ai.strategy.estimators import EmpiricalEstimator, ParametricEstimator
from pitwall_ai.strategy.scenarios import Scenario
from pitwall_ai.strategy.simulator import (
    SimulationConfig,
    Stint,
    Strategy,
    monte_carlo,
    simulate_strategy,
    tyre_profile,
)


@pytest.fixture
def estimator():
    """lap = 90 + offset + deg * (age - 1); no fuel effect, so numbers are easy to check."""
    return ParametricEstimator(
        90.0,
        {"SOFT": -1.0, "MEDIUM": 0.0, "HARD": 1.0},
        {"SOFT": 0.2, "MEDIUM": 0.1, "HARD": 0.05},
        fuel_gain_per_lap=0.0,
    )


ONE_STOP = Strategy((Stint("MEDIUM", 3), Stint("HARD", 2)), "one-stop")


def test_remaining_time_and_pit_cost_exact(estimator):
    result = simulate_strategy(ONE_STOP, estimator, SimulationConfig(5, pit_loss_seconds=20.0))
    # MEDIUM ages 1-3: 90, 90.1, 90.2 ; HARD ages 1-2: 91, 91.05
    assert result.driving_time == pytest.approx(452.35)
    assert result.pit_cost == pytest.approx(20.0)  # counted exactly once
    assert result.total_time == pytest.approx(472.35)
    assert result.pit_laps == [3]


def test_pit_cost_scales_with_number_of_stops_and_zero_stops(estimator):
    two_stop = Strategy((Stint("SOFT", 2), Stint("MEDIUM", 2), Stint("HARD", 1)))
    no_stop = Strategy((Stint("MEDIUM", 5),))
    cfg = SimulationConfig(5, pit_loss_seconds=20.0)
    assert simulate_strategy(two_stop, estimator, cfg).pit_cost == pytest.approx(40.0)
    assert simulate_strategy(no_stop, estimator, cfg).pit_cost == 0.0


def test_tyre_age_resets_after_pit_stop():
    _, ages = tyre_profile(ONE_STOP)
    assert ages.tolist() == [1, 2, 3, 1, 2]
    _, ages = tyre_profile(ONE_STOP, start_tyre_age=5)
    assert ages.tolist() == [5, 6, 7, 1, 2]


def test_stint_lengths_must_sum_to_remaining_laps(estimator):
    with pytest.raises(ValueError, match="add up to 5"):
        simulate_strategy(ONE_STOP, estimator, SimulationConfig(6))


def test_pit_lap_numbers_use_race_lap_offsets():
    assert ONE_STOP.pit_laps(start_lap=1) == [3]
    assert ONE_STOP.pit_laps(start_lap=21) == [23]


@pytest.mark.parametrize(
    ("strategy", "kwargs"),
    [
        (Strategy(()), {}),
        (Strategy((Stint("MEDIUM", 0), Stint("HARD", 5))), {}),
        (Strategy((Stint("MEDIUM", 5),)), {"allowed_compounds": ("SOFT",)}),
        (Strategy((Stint("MEDIUM", 5),)), {"require_two_compounds": True}),
        (Strategy((Stint("MEDIUM", 2), Stint("HARD", 3))), {"min_stint": 3}),
    ],
)
def test_invalid_strategies_rejected(strategy, kwargs):
    with pytest.raises(ValueError):
        strategy.validate(5, **kwargs)


def test_invalid_config_rejected(estimator):
    for cfg in (
        SimulationConfig(0),
        SimulationConfig(5, pit_loss_seconds=-1),
        SimulationConfig(5, start_tyre_age=0),
    ):
        with pytest.raises(ValueError):
            simulate_strategy(ONE_STOP, estimator, cfg)


def test_unknown_compound_in_estimator_is_reported(estimator):
    bad = Strategy((Stint("WET", 5),))
    with pytest.raises(ValueError, match="WET"):
        simulate_strategy(bad, estimator, SimulationConfig(5))


def test_scenario_degradation_scale_and_pace_offset(estimator):
    base = simulate_strategy(ONE_STOP, estimator, SimulationConfig(5, 20.0))
    scaled = Scenario(degradation_scale=2.0, pace_offset_seconds=1.0)
    result = simulate_strategy(ONE_STOP, estimator, SimulationConfig(5, 20.0, scenario=scaled))
    # degradation part: 0+0.1+0.2+0+0.05 = 0.35 s -> doubled; plus 1 s on each of 5 laps
    assert result.driving_time == pytest.approx(base.driving_time + 0.35 + 5.0)


def test_safety_car_slows_laps_and_discounts_pit_stop(estimator):
    sc = Scenario(
        safety_car_start_lap=3,
        safety_car_laps=1,
        safety_car_lap_time_factor=1.5,
        safety_car_pit_loss_factor=0.5,
    )
    result = simulate_strategy(ONE_STOP, estimator, SimulationConfig(5, 20.0, scenario=sc))
    assert result.pit_costs == [10.0]  # stop at end of lap 3, inside the SC window
    assert result.lap_times[2] == pytest.approx(90.2 * 1.5)
    assert result.lap_times[0] == pytest.approx(90.0)


def test_scenario_validation():
    for bad in (
        Scenario(degradation_scale=-1),
        Scenario(safety_car_lap_time_factor=0.5),
        Scenario(safety_car_start_lap=0, safety_car_laps=2),
        Scenario(safety_car_pit_loss_factor=2),
    ):
        with pytest.raises(ValueError):
            bad.validate()


def test_monte_carlo_is_reproducible_and_consistent(estimator):
    other = Strategy((Stint("SOFT", 2), Stint("HARD", 3)), "soft-hard")
    cfg = SimulationConfig(5, 20.0)
    a = monte_carlo([ONE_STOP, other], estimator, cfg, n_runs=200, seed=7)
    b = monte_carlo([ONE_STOP, other], estimator, cfg, n_runs=200, seed=7)
    c = monte_carlo([ONE_STOP, other], estimator, cfg, n_runs=200, seed=8)
    assert a.equals(b) and not a.equals(c)
    assert a["FractionOfRunsBest"].sum() == pytest.approx(1.0)
    assert (a["P10"] <= a["P90"]).all()


def test_monte_carlo_with_zero_uncertainty_matches_deterministic(estimator):
    cfg = SimulationConfig(5, 20.0)
    out = monte_carlo(
        [ONE_STOP],
        estimator,
        cfg,
        n_runs=5,
        degradation_sigma=0.0,
        compound_pace_sigma=0.0,
        pit_loss_sigma=0.0,
    )
    assert out["MeanTime"].iloc[0] == pytest.approx(
        simulate_strategy(ONE_STOP, estimator, cfg).total_time
    )
    assert out["StdTime"].iloc[0] == pytest.approx(0.0)


def test_monte_carlo_input_validation(estimator):
    cfg = SimulationConfig(5)
    with pytest.raises(ValueError):
        monte_carlo([], estimator, cfg)
    with pytest.raises(ValueError):
        monte_carlo([ONE_STOP], estimator, cfg, n_runs=0)
    with pytest.raises(ValueError, match="unique"):
        monte_carlo([ONE_STOP, ONE_STOP], estimator, cfg)


def test_parametric_estimator_fuel_effect_uses_race_lap():
    est = ParametricEstimator(90.0, {"MEDIUM": 0.0}, {"MEDIUM": 0.0}, fuel_gain_per_lap=0.1)
    assert est.predict(["MEDIUM"], np.array([1]), np.array([11]))[0] == pytest.approx(89.0)
    with pytest.raises(ValueError):
        ParametricEstimator(-1, {"MEDIUM": 0}, {"MEDIUM": 0}, 0)


def test_empirical_estimator_recovers_demo_degradation(demo_clean):
    est = EmpiricalEstimator.fit(demo_clean, base_pace=90.0)
    # Synthetic truth: SOFT degrades 0.09 s/lap, HARD 0.035 s/lap.
    assert est.slopes["SOFT"] == pytest.approx(0.09, abs=0.02)
    assert est.slopes["HARD"] == pytest.approx(0.035, abs=0.02)
    assert est.slopes["SOFT"] > est.slopes["HARD"]
    with pytest.raises(ValueError):
        EmpiricalEstimator.fit(demo_clean.head(5), 90.0)
