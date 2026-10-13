"""
Result storage (src/evaluation/results.py).

One CSV row per run, written the moment the run finishes, so a night that
stops early keeps everything computed so far. Each row carries a unique key
built from its factors; the runner skips keys that are already present, which
makes a restart continue instead of repeating work.

When the set of columns grows (new metrics), an existing file is migrated on
opening: its rows are kept and the new columns stay empty for them.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, fields

RUN_KEY_FIELDS = (
    "experiment",
    "process",
    "cases",
    "capacities",
    "level",
    "method",
    "publish",
    "n_scenarios",
    "visibility_weight",
    "seed",
    "time_limit",
    "objective",
    "sla_factor",
)

#: How a plan's start times were set.
PUBLISH_NONE = "none"          # no start times promised (all zero)
PUBLISH_OPTIMISED = "optimised"  # set by the model through the weight lambda
PUBLISH_OWN = "own"            # a baseline's own planning-scenario schedule


def default_publish(method: str, visibility_weight: float) -> str:
    """The publication mode of rows written before the column existed."""
    if method in ("det", "rob"):
        return PUBLISH_OWN
    return PUBLISH_OPTIMISED if visibility_weight > 0 else PUBLISH_NONE


@dataclass(frozen=True)
class RunRow:
    """Everything recorded about one run."""

    # factors
    experiment: str
    process: str
    cases: int
    capacities: str
    level: str
    method: str
    n_scenarios: int
    visibility_weight: float
    seed: int
    # solver outcome
    status: str
    objective_in_sample: float | None
    bound: float | None
    gap: float | None
    wall_seconds: float
    deterministic_seconds: float
    n_variables: int
    n_constraints: int
    # out-of-sample behaviour of the published plan
    oos_makespan: float | None
    oos_makespan_std: float | None
    oos_visibility: float | None
    oos_mean_delay: float | None
    oos_max_delay: int | None
    note: str = ""
    # added with the visibility study
    publish: str = PUBLISH_NONE
    oos_visibility_5: float | None = None
    oos_visibility_15: float | None = None
    oos_on_time_share: float | None = None
    oos_on_time_share_5: float | None = None
    oos_on_time_share_15: float | None = None
    plan_file: str = ""
    # empty when the run used the command line default
    time_limit: str = ""
    # added with the value-of-the-stochastic-solution study
    objective: str = "makespan"
    sla_factor: str = ""
    due_date: int | None = None
    oos_mean_case_completion: float | None = None
    oos_total_tardiness: float | None = None
    oos_sla_violation: float | None = None

    @property
    def key(self) -> str:
        values = asdict(self)
        return json.dumps({name: values[name] for name in RUN_KEY_FIELDS}, sort_keys=True)


class ResultsFile:
    """Append-only CSV of RunRow records with a resume check."""

    def __init__(self, path: str):
        self._path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fieldnames = [field.name for field in fields(RunRow)]
        self._migrate_if_needed()
        self._done = self._load_keys()

    @property
    def path(self) -> str:
        return self._path

    def __contains__(self, key: str) -> bool:
        return key in self._done

    def append(self, row: RunRow) -> None:
        exists = os.path.exists(self._path) and os.path.getsize(self._path) > 0
        with open(self._path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(asdict(row))
            handle.flush()
            os.fsync(handle.fileno())
        self._done.add(row.key)

    def rows(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        with open(self._path, newline="") as handle:
            return list(csv.DictReader(handle))

    def _migrate_if_needed(self) -> None:
        if not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
            return
        with open(self._path, newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            rows = list(reader)
        if header == self._fieldnames:
            return
        for row in rows:
            if not row.get("objective"):
                row["objective"] = "makespan"
            if not row.get("publish"):
                row["publish"] = default_publish(
                    row["method"], float(row["visibility_weight"])
                )
        temporary = self._path + ".migrating"
        with open(temporary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in self._fieldnames})
        os.replace(temporary, self._path)

    def _load_keys(self) -> set[str]:
        return {
            json.dumps({name: _restore(name, row[name]) for name in RUN_KEY_FIELDS}, sort_keys=True)
            for row in self.rows()
        }


def _restore(name: str, value: str):
    """CSV stores strings; the key fields need their original types back."""
    if name in ("cases", "n_scenarios", "seed"):
        return int(value)
    if name == "visibility_weight":
        return float(value)
    return value
