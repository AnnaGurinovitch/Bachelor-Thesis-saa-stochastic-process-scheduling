"""
Analysis of the experiment results (scripts/analyse_results.py).

    python scripts/analyse_results.py                  # reads results/*.csv
    python scripts/analyse_results.py --results other_dir

Writes tables (CSV and Markdown) and figures (PNG) to <results>/analysis/.
Every section is skipped quietly if its experiment has not been run yet, so the
script can be used while experiments are still running.

Sections
  E1 / E1b  convergence in N and value of SAA against DET and ROB
  E2        tractability of the two formulations
  E3 / E3b  price of visibility: lambda versus published quantiles
  E4        main evaluation: value of SAA, borders of visibility, solve status
  E5        does the tractability boundary move with more time?
  E6        value of the stochastic solution: mixed-risk cases and deadlines
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SOLVED = ("OPTIMAL", "FEASIBLE")
QUANTILES = ("q50", "q70", "q80", "q90", "q95")
#: on-time share a plan must reach to count as "visible" in the border analysis
VISIBILITY_TARGET = 0.8
LEVEL_COLOURS = {"D1": "#4C78A8", "D2": "#F58518", "D3": "#E45756"}
METHOD_NAMES = {"saa_cp": "SAA-CP", "saa_mseqct": "SAA-MSEQCT", "det": "DET", "rob": "ROB"}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class Report:
    """Collects Markdown sections and writes tables and figures."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.sections: list[str] = ["# Experiment results\n"]

    def table(self, name: str, frame: pd.DataFrame, title: str, note: str = "") -> None:
        frame.to_csv(os.path.join(self.directory, f"{name}.csv"), index=False)
        self.sections.append(f"## {title}\n")
        if note:
            self.sections.append(note + "\n")
        self.sections.append(_markdown(frame) + "\n")

    def figure(self, name: str, fig, title: str) -> None:
        path = os.path.join(self.directory, f"{name}.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        self.sections.append(f"**Figure:** {title} (`{name}.png`)\n")

    def write(self) -> str:
        path = os.path.join(self.directory, "summary.md")
        with open(path, "w") as handle:
            handle.write("\n".join(self.sections))
        return path


def _markdown(frame: pd.DataFrame) -> str:
    """A pipe table without requiring the optional 'tabulate' package."""
    def fmt(value):
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.3g}" if abs(value) < 1000 else f"{value:.0f}"
        return str(value)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def load(results: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(results, f"{name}.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, keep_default_na=True)
    if frame.empty:
        return None
    if "publish" not in frame:
        frame["publish"] = "none"
    frame["publish"] = frame["publish"].fillna("none")
    frame["solved"] = frame["status"].isin(SOLVED)
    return frame


def solved(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["solved"]]


def gain_percent(reference: float, value: float) -> float:
    """How much lower `value` is than `reference`, in percent."""
    return 100.0 * (reference - value) / reference


# ----------------------------------------------------------------------
# E1 / E1b
# ----------------------------------------------------------------------
def analyse_convergence(report: Report, frame: pd.DataFrame, name: str) -> None:
    ok = solved(frame)
    saa = ok[(ok["method"] == "saa_cp") & (ok["publish"] == "none")]
    base = ok[ok["method"].isin(["det", "rob"]) & (ok["publish"] == "none")]
    if saa.empty:
        return

    curve = (
        saa.groupby(["level", "n_scenarios"])
        .agg(makespan=("oos_makespan", "mean"), sd=("oos_makespan", "std"),
             optimal=("status", lambda s: (s == "OPTIMAL").mean()),
             seconds=("wall_seconds", "mean"))
        .reset_index()
    )
    report.table(f"{name}_convergence", curve.round(3),
                 f"{name}: out-of-sample makespan of SAA by N",
                 "Mean and standard deviation over seeds; `optimal` is the share of runs "
                 "solved to optimality.")

    rows = []
    for level, group in curve.groupby("level"):
        best = group.loc[group["makespan"].idxmin()]
        entry = {"level": level, "saa_best": best["makespan"], "best_n": int(best["n_scenarios"])}
        for method in ("det", "rob"):
            value = base[(base["level"] == level) & (base["method"] == method)]["oos_makespan"]
            if not value.empty:
                entry[method] = value.mean()
                entry[f"gain_vs_{method}_%"] = gain_percent(value.mean(), best["makespan"])
        rows.append(entry)
    report.table(f"{name}_value_of_saa", pd.DataFrame(rows).round(2),
                 f"{name}: value of SAA against the baselines (all without start times)")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for level, group in curve.groupby("level"):
        colour = LEVEL_COLOURS.get(level, None)
        ax.errorbar(group["n_scenarios"], group["makespan"], yerr=group["sd"].fillna(0),
                    marker="o", capsize=3, label=f"SAA {level}", color=colour)
        det = base[(base["level"] == level) & (base["method"] == "det")]["oos_makespan"]
        if not det.empty:
            ax.axhline(det.mean(), linestyle="--", color=colour, alpha=0.6, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(sorted(curve["n_scenarios"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of scenarios N")
    ax.set_ylabel("expected makespan, out-of-sample (min)")
    ax.set_title(f"{name}: convergence (dashed: DET)")
    ax.legend(frameon=False)
    report.figure(f"{name}_convergence", fig, f"{name}: SAA makespan against N, DET dashed")


# ----------------------------------------------------------------------
# E2
# ----------------------------------------------------------------------
def analyse_tractability(report: Report, frame: pd.DataFrame) -> None:
    table = (
        frame.groupby(["cases", "process", "capacities", "n_scenarios", "method"])
        .agg(optimal=("status", lambda s: (s == "OPTIMAL").mean()),
             solved=("solved", "mean"),
             gap=("gap", "mean"),
             seconds=("wall_seconds", "mean"),
             makespan=("oos_makespan", "mean"),
             variables=("n_variables", "mean"))
        .reset_index()
    )
    report.table("E2_tractability", table.round(3), "E2: tractability per instance and formulation",
                 "`optimal`: share of seeds solved to optimality; `gap`: mean relative gap at "
                 "termination; `makespan`: out-of-sample.")

    wide = table.pivot_table(index=["cases", "process", "capacities", "n_scenarios"],
                             columns="method", values="makespan").reset_index()
    if {"saa_cp", "saa_mseqct"} <= set(wide.columns):
        wide["mseqct_minus_cp_%"] = 100 * (wide["saa_mseqct"] - wide["saa_cp"]) / wide["saa_cp"]
        report.table("E2_formulation_quality", wide.round(2),
                     "E2: plan quality of the two formulations",
                     "Positive values: the MSEQCT plan is worse (longer) than the CP plan.")

    by_size = (frame.groupby(["cases", "method", "n_scenarios"])["status"]
               .apply(lambda s: (s == "OPTIMAL").mean()).reset_index(name="optimal"))
    fig, axes = plt.subplots(1, by_size["n_scenarios"].nunique(), figsize=(9, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (n, group) in zip(axes, by_size.groupby("n_scenarios")):
        for method, line in group.groupby("method"):
            ax.plot(line["cases"], line["optimal"], marker="o", label=METHOD_NAMES.get(method, method))
        ax.set_title(f"N = {n}")
        ax.set_xlabel("concurrent cases")
        ax.set_xticks(sorted(group["cases"].unique()))
    axes[0].set_ylabel("share solved to optimality")
    axes[0].legend(frameon=False)
    report.figure("E2_optimal_share", fig, "E2: share of runs solved to optimality by size")


# ----------------------------------------------------------------------
# E3 / E3b
# ----------------------------------------------------------------------
def analyse_visibility_tradeoff(report: Report, frame: pd.DataFrame, name: str) -> None:
    ok = solved(frame)
    if "oos_on_time_share" not in ok or ok["oos_on_time_share"].isna().all():
        report.sections.append(
            f"_{name}: no on-time share recorded (run before the visibility metrics existed)._\n"
        )
        return

    def label(row) -> str:
        if row["method"] in ("det", "rob"):
            return f"{METHOD_NAMES[row['method']]} ({row['publish']})"
        if row["publish"] == "optimised":
            return f"lambda={row['visibility_weight']:g}"
        if row["publish"] == "none":
            return "SAA, no times"
        return f"SAA {row['publish']}"

    ok = ok.assign(strategy=ok.apply(label, axis=1))
    table = (
        ok.groupby(["process", "strategy"])
        .agg(makespan=("oos_makespan", "mean"), on_time=("oos_on_time_share", "mean"),
             on_time_15=("oos_on_time_share_15", "mean"), visible=("oos_visibility", "mean"),
             visible_15=("oos_visibility_15", "mean"), seconds=("wall_seconds", "mean"))
        .reset_index()
        .sort_values(["process", "makespan"])
    )
    report.table(f"{name}_tradeoff", table.round(3),
                 f"{name}: makespan against visibility per strategy",
                 "`on_time`: share of tasks starting as planned; `visible`: share of scenarios in "
                 "which every task starts as planned; `_15`: with 15 minutes tolerance.")

    processes = sorted(table["process"].unique())
    fig, axes = plt.subplots(1, len(processes), figsize=(5.2 * len(processes), 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, process in zip(axes, processes):
        part = table[table["process"] == process]
        for kind, marker, colour in (
            ("SAA q", "o", "#4C78A8"), ("lambda", "s", "#F58518"),
            ("DET", "^", "#54A24B"), ("ROB", "v", "#E45756"),
        ):
            points = part[part["strategy"].str.startswith(kind)]
            if kind == "SAA q":
                points = points.sort_values("on_time")
                ax.plot(points["on_time"], points["makespan"], marker=marker, color=colour,
                        label="SAA + quantile")
            elif kind == "lambda":
                points = points.sort_values("on_time")
                ax.plot(points["on_time"], points["makespan"], marker=marker, color=colour,
                        label="SAA + lambda")
            else:
                points = points[points["strategy"].str.contains("own")]
                ax.scatter(points["on_time"], points["makespan"], marker=marker, color=colour,
                           s=60, label=f"{kind} (own times)", zorder=3)
        ax.set_title(process)
        ax.set_xlabel("share of tasks starting on time")
        ax.set_ylabel("expected makespan (min)")
        ax.set_xlim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)
    report.figure(f"{name}_tradeoff", fig, f"{name}: price of visibility (lower right is better)")


# ----------------------------------------------------------------------
# E4
# ----------------------------------------------------------------------
def analyse_main(report: Report, frame: pd.DataFrame) -> None:
    ok = solved(frame)
    keys = ["process", "cases", "capacities", "level"]

    # --- solve status by size
    saa_none = frame[(frame["method"] == "saa_cp") & (frame["publish"] == "none")]
    status = (saa_none.groupby(["cases", "process"])
              .agg(runs=("status", "size"),
                   optimal=("status", lambda s: (s == "OPTIMAL").mean()),
                   solved=("solved", "mean"), gap=("gap", "mean"),
                   seconds=("wall_seconds", "mean"))
              .reset_index())
    report.table("E4_solve_status", status.round(3), "E4: solve status by size")

    # --- value of SAA against DET, without start times
    saa = ok[(ok["method"] == "saa_cp") & (ok["publish"] == "none")].groupby(keys)["oos_makespan"].mean()
    det = ok[(ok["method"] == "det") & (ok["publish"] == "none")].groupby(keys)["oos_makespan"].mean()
    rob = ok[(ok["method"] == "rob") & (ok["publish"] == "none")].groupby(keys)["oos_makespan"].mean()
    value = pd.concat({"saa": saa, "det": det, "rob": rob}, axis=1).dropna(subset=["saa", "det"])
    value["gain_vs_det_%"] = gain_percent(value["det"], value["saa"])
    value["gain_vs_rob_%"] = gain_percent(value["rob"], value["saa"])
    value = value.reset_index()
    summary = (value.groupby(["level", "cases"])[["gain_vs_det_%", "gain_vs_rob_%"]]
               .mean().reset_index())
    report.table("E4_value_of_saa", summary.round(2),
                 "E4: how much shorter the SAA plan is than DET and ROB (mean over instances)",
                 "All three without start times, so only modes and order are compared.")

    # --- borders of visibility
    saa_all = ok[ok["method"] == "saa_cp"]
    per_pub = (saa_all.groupby(keys + ["publish"])
               .agg(makespan=("oos_makespan", "mean"), on_time=("oos_on_time_share", "mean"),
                    on_time_15=("oos_on_time_share_15", "mean"))
               .reset_index())
    base = per_pub[per_pub["publish"] == "none"][keys + ["makespan"]].rename(
        columns={"makespan": "makespan_no_times"})
    quantiles = per_pub[per_pub["publish"].isin(QUANTILES)].merge(base, on=keys)
    quantiles["cost_%"] = 100 * (quantiles["makespan"] - quantiles["makespan_no_times"]) \
        / quantiles["makespan_no_times"]

    def border(group: pd.DataFrame) -> pd.Series:
        reached = group[group["on_time"] >= VISIBILITY_TARGET]
        if reached.empty:
            return pd.Series({"reached": False, "cheapest": "", "cost_%": np.nan,
                              "best_on_time": group["on_time"].max()})
        cheapest = reached.loc[reached["cost_%"].idxmin()]
        return pd.Series({"reached": True, "cheapest": cheapest["publish"],
                          "cost_%": cheapest["cost_%"], "best_on_time": group["on_time"].max()})

    borders = quantiles.groupby(keys).apply(border, include_groups=False).reset_index()
    report.table("E4_visibility_borders", borders.round(3),
                 f"E4: cheapest publication reaching {VISIBILITY_TARGET:.0%} on-time tasks",
                 "`cost_%`: makespan increase over the same plan without start times. "
                 "`reached = False` marks instances beyond the border of visibility.")

    reach = (borders.groupby(["level", "cases"])
             .agg(share_reached=("reached", "mean"), median_cost_pct=("cost_%", "median"))
             .reset_index())
    report.table("E4_visibility_summary", reach.round(3),
                 "E4: share of instances where the visibility target is reachable")

    # heatmap: cost of visibility, one panel per level
    levels = sorted(borders["level"].unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        part = borders[borders["level"] == level].copy()
        part["row"] = part["process"] + " " + part["capacities"].str.replace("Junior", "J") \
            .str.replace("Senior", "S")
        grid = part.pivot_table(index="row", columns="cases", values="cost_%")
        reached = part.pivot_table(index="row", columns="cases", values="reached", aggfunc="max")
        image = ax.imshow(grid.values, cmap="viridis_r", aspect="auto", vmin=0,
                          vmax=np.nanmax(borders["cost_%"].values) if borders["cost_%"].notna().any() else 1)
        for (i, j), cost in np.ndenumerate(grid.values):
            flag = reached.values[i, j]
            if np.isnan(cost) and (flag is None or (isinstance(flag, float) and np.isnan(flag))):
                continue  # combination not run
            text = "×" if not bool(flag) else f"{cost:.0f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if not np.isnan(cost) and cost > 0.5 * np.nanmax(grid.values) else "black")
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index, fontsize=7)
        ax.set_xlabel("concurrent cases")
        ax.set_title(level)
    fig.colorbar(image, ax=axes, shrink=0.8, label="makespan cost of visibility (%)")
    report.figure("E4_visibility_borders", fig,
                  f"E4: cost of reaching {VISIBILITY_TARGET:.0%} on-time tasks; × = not reachable")

    # trade-off averaged over all instances of a level: quantiles versus baselines
    points = (ok.groupby(["level", "method", "publish"])
              .agg(makespan=("oos_makespan", "mean"), on_time=("oos_on_time_share", "mean"),
                   on_time_15=("oos_on_time_share_15", "mean"))
              .reset_index())
    report.table("E4_tradeoff_by_level", points.round(3),
                 "E4: makespan and on-time share per strategy, mean over all instances")
    fig, axes = plt.subplots(1, len(levels), figsize=(4.4 * len(levels), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        part = points[points["level"] == level]
        curve = part[(part["method"] == "saa_cp") & part["publish"].isin(QUANTILES)] \
            .sort_values("on_time")
        ax.plot(curve["on_time"], curve["makespan"], marker="o", color="#4C78A8",
                label="SAA + quantile $q$")
        for _, row in curve.iterrows():
            ax.annotate(row["publish"], (row["on_time"], row["makespan"]), fontsize=7,
                        xytext=(4, -8), textcoords="offset points")
        for method, marker, colour, name in (("det", "^", "#54A24B", "DET"),
                                             ("rob", "v", "#E45756", "ROB")):
            own = part[(part["method"] == method) & (part["publish"] == "own")]
            ax.scatter(own["on_time"], own["makespan"], marker=marker, s=70, color=colour,
                       zorder=3, label=f"{name} (own times)")
        none = part[(part["method"] == "saa_cp") & (part["publish"] == "none")]
        if not none.empty:
            ax.axhline(none["makespan"].iloc[0], linestyle=":", color="gray", linewidth=1,
                       label="SAA, no times")
        ax.set_title(level)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of tasks starting on time")
    axes[0].set_ylabel("expected makespan (min)")
    axes[0].legend(frameon=False, fontsize=7)
    report.figure("E4_tradeoff_by_level", fig,
                  "E4: price of visibility per variability level, mean over all instances")


# ----------------------------------------------------------------------
# E5
# ----------------------------------------------------------------------
def analyse_time_limits(report: Report, frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["time_limit"] = pd.to_numeric(frame["time_limit"], errors="coerce")
    table = (frame.groupby(["process", "cases", "method", "time_limit"])
             .agg(status=("status", "first"), gap=("gap", "mean"),
                  makespan=("oos_makespan", "mean"), seconds=("wall_seconds", "mean"))
             .reset_index())
    report.table("E5_time_limits", table.round(3), "E5: effect of the time limit")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (process, cases, method), group in table.groupby(["process", "cases", "method"]):
        ax.plot(group["time_limit"], group["gap"], marker="o",
                label=f"{process}, {cases} cases, {METHOD_NAMES.get(method, method)}")
    ax.set_xscale("log")
    ax.set_xlabel("time limit (s)")
    ax.set_ylabel("relative gap at termination")
    ax.legend(frameon=False, fontsize=7)
    report.figure("E5_gap_by_time_limit", fig, "E5: gap against time limit")


# ----------------------------------------------------------------------
# E6
# ----------------------------------------------------------------------
def analyse_value_of_stochastic_solution(report: Report, frame: pd.DataFrame) -> None:
    ok = solved(frame).copy()
    ok["sla_factor"] = ok["sla_factor"].fillna("").astype(str)
    keys = ["process", "cases", "capacities", "level", "objective", "sla_factor"]
    metrics = dict(makespan=("oos_makespan", "mean"), tardiness=("oos_total_tardiness", "mean"),
                   violation=("oos_sla_violation", "mean"),
                   completion=("oos_mean_case_completion", "mean"))
    per = ok.groupby(keys + ["method"]).agg(**metrics).reset_index()
    wide = per.pivot_table(index=keys, columns="method",
                           values=["makespan", "tardiness", "violation", "completion"])
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()

    def vss(row, metric):
        det, saa = row.get(f"{metric}_det"), row.get(f"{metric}_saa_cp")
        if pd.isna(det) or pd.isna(saa) or det == 0:
            return np.nan
        return 100.0 * (det - saa) / det

    wide["vss_makespan_%"] = wide.apply(lambda r: vss(r, "makespan"), axis=1)
    wide["vss_tardiness_%"] = wide.apply(lambda r: vss(r, "tardiness"), axis=1)
    wide["violation_reduction_pp"] = 100 * (wide.get("violation_det") - wide.get("violation_saa_cp")) \
        if "violation_det" in wide else np.nan
    report.table("E6_per_instance", wide.round(3),
                 "E6: SAA against DET and ROB per instance",
                 "`vss_*`: value of the stochastic solution, how much lower SAA is than DET "
                 "(percent); `violation_reduction_pp`: fewer SLA violations in percentage points.")

    summary = (wide.groupby(["objective", "sla_factor", "level", "cases"])
               [["vss_makespan_%", "vss_tardiness_%", "violation_reduction_pp",
                 "violation_det", "violation_saa_cp"]]
               .mean().reset_index())
    report.table("E6_summary", summary.round(2),
                 "E6: value of the stochastic solution by objective, risk profile and size")

    sla = summary[summary["objective"] == "sla"]
    if not sla.empty:
        factors = sorted(sla["sla_factor"].unique())
        fig, axes = plt.subplots(1, len(factors), figsize=(5.2 * len(factors), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, factor in zip(axes, factors):
            part = sla[sla["sla_factor"] == factor]
            levels = sorted(part["level"].unique())
            width = 0.8 / max(len(levels), 1)
            for i, level in enumerate(levels):
                bars = part[part["level"] == level]
                ax.bar(bars["cases"] + (i - (len(levels) - 1) / 2) * width,
                       bars["vss_tardiness_%"], width=width, label=level)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"due date = {factor} x reference")
            ax.set_xlabel("concurrent cases")
            ax.set_xticks(sorted(part["cases"].unique()))
        axes[0].set_ylabel("tardiness reduction SAA vs DET (%)")
        axes[0].legend(frameon=False)
        report.figure("E6_value_of_stochastic_solution", fig,
                      "E6: how much less tardiness the SAA plan has than the mean-based plan")


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args(argv)

    report = Report(os.path.join(args.results, "analysis"))
    sections = [
        ("E1", lambda f: analyse_convergence(report, f, "E1")),
        ("E1b", lambda f: analyse_convergence(report, f, "E1b")),
        ("E2", lambda f: analyse_tractability(report, f)),
        ("E3", lambda f: analyse_visibility_tradeoff(report, f, "E3")),
        ("E3b", lambda f: analyse_visibility_tradeoff(report, f, "E3b")),
        ("E4", lambda f: analyse_main(report, f)),
        ("E5", lambda f: analyse_time_limits(report, f)),
        ("E6", lambda f: analyse_value_of_stochastic_solution(report, f)),
    ]
    for name, analyse in sections:
        frame = load(args.results, name)
        if frame is None:
            print(f"{name}: no results yet, skipped")
            continue
        print(f"{name}: {len(frame)} rows")
        analyse(frame)

    print(f"summary written to {report.write()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
