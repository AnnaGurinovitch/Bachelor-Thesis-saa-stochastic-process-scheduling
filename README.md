# Bachelor-Thesis-saa-stochastic-process-scheduling

Sample Average Approximation for stochastic business process scheduling
(bachelor thesis, TUM).

Two SAA formulations of the same proactive scheduling problem are compared,
both solved with Google CP-SAT:

* `saa_cp` — interval variables and cumulative resource constraints
* `saa_mseqct` — sequence-based continuous-time model (Klein 2025)

A plan fixes the execution mode of every task, the order in which tasks claim
their resource pool, and the planned start times that are published to the
process participants. Plans are judged out-of-sample: they are executed on
1,000 hold-out scenarios by the simulator in `saa_scheduling/execution`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Running experiments

| Experiment | Question | Command |
| --- | --- | --- |
| `smoke` | does everything run? (minutes) | `run_experiments.py smoke` |
| `E1`, `E1b` | convergence in N, value of SAA (easy / hard instance) | `run_experiments.py E1` |
| `E2` | tractability of the two formulations | `run_experiments.py E2` |
| `E3`, `E3b` | price of visibility: lambda versus published quantiles | `run_experiments.py E3b` |
| `E4` | main evaluation: full complexity grid, three seeds | `run_experiments.py E4 --seconds 120` |
| `E5` | does the tractability boundary move with more time? | `run_experiments.py E5` |
| `E6` | where the mean is blind: mixed-risk cases and due dates | `run_experiments.py E6 --seconds 120` |

All commands are run as `.venv/bin/python scripts/<command>`. Results are
appended to `results/<experiment>.csv` after every single run, every plan is
stored under `results/plans/`, and re-running the same command continues where
the last one stopped. For an overnight run keep the machine awake:

```bash
caffeinate -i .venv/bin/python scripts/run_experiments.py E4 --seconds 120
```

## Analysis

```bash
.venv/bin/python scripts/analyse_results.py
```

Reads every `results/*.csv` that exists (also while experiments are still
running) and writes tables, figures and a `summary.md` to `results/analysis/`.

## Verification of the results

```
.venv/bin/python scripts/verify_plans.py --sample 0
```

Loads every stored plan, rebuilds the instance and the hold-out sample and
runs the simulator again, then compares the result with the numbers in the
CSV files. It needs no solver and reports any row whose recorded metrics
cannot be reproduced from its plan.

## Layout

| Path | Contents |
| --- | --- |
| `src/saa_scheduling/basics` | process models P1–P3, modes, problem instances |
| `src/saa_scheduling/distributions` | duration distributions and the levels D1–D3 |
| `src/saa_scheduling/sampling` | scenario generation |
| `src/saa_scheduling/execution` | plan, execution semantics, metrics |
| `src/saa_scheduling/solvers` | the two CP-SAT formulations |
| `src/evaluation` | baselines, experiment definitions, runner, results file |
| `scripts/run_experiments.py` | command line entry point |
| `scripts/analyse_results.py` | tables and figures of the evaluation |
| `scripts/verify_plans.py` | re-simulates the stored plans and checks the CSV rows |
| `Thesis Title/` | the LaTeX thesis |
