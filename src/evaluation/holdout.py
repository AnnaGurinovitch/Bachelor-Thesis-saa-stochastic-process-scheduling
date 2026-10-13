"""
Hold-out sample (src/evaluation/holdout.py).

Every plan is evaluated on the same independent scenarios: one sample per
instance, drawn with a seed that no optimisation run uses. Keeping this in
its own module lets the runner and the verification script share it without
importing a solver.
"""

from __future__ import annotations

from saa_scheduling.basics.instances import build_instance
from saa_scheduling.distributions.duration_profiles import build_case_models
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator

#: Hold-out sample: independent of every optimisation seed.
HOLDOUT_SEED = 900_001
HOLDOUT_SIZE = 1000


def holdout_scenarios(
    process: str,
    cases: int,
    capacities: dict[str, int],
    level: str,
    size: int = HOLDOUT_SIZE,
) -> tuple:
    """The hold-out sample of one instance."""
    instance = build_instance(process, cases, capacities)
    generator = ScenarioGenerator(instance, build_case_models(level, cases), seed=HOLDOUT_SEED)
    return generator.generate(size)
