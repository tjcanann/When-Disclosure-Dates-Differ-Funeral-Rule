#!/usr/bin/env python3
"""Phase 3D diagnostics for the CEP funeral-price paper.

Input
-----
analytic_panel.csv : locked, source-audited establishment-year panel.

Outputs
-------
phase3d_output/*.csv covering:
  1. Economic magnitude of same-GPL-date repricing.
  2. Lead/lag audit of adjacent GPL-date changes.
  3. Information/signal metrics for the GPL-date-change indicator.
  4. Wild-cluster bootstrap and leave-one-out inference checks.

The script is deterministic. Wild-cluster bootstrap uses Rademacher weights at
the establishment level and restricted residuals under H0: beta=0.
"""
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
PANEL = HERE / "analytic_panel.csv"
OUT = HERE / "phase3d_output"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402

SEED = 20260827
B_WILD = 4999
B_PERM_LOO = 1999

df = pd.read_csv(PANEL)
main = df[
    df["gap_years_clean"].eq(1)
    & df["n_headline_comparable_clean"].ge(2)
    & df["formal_revision_clean"].notna()
    & df["any_headline_change_clean"].notna()
].copy()
main["prior_gpl_age_years"] = main["prev_gpl_age_days_clean"] / 365.25

# -----------------------------------------------------------------------------
# 1. Economic magnitude of same-date repricing
# -----------------------------------------------------------------------------
components = [
    ("direct_cremation", "prev_direct_cremation_clean", "direct_cremation", "delta_direct_cremation_clean", "price_change_direct_cremation_clean"),
    ("immediate_burial", "prev_immediate_burial_clean", "immediate_burial", "delta_immediate_burial_clean", "price_change_immediate_burial_clean"),
    ("basic_services_fee", "prev_ndf_clean", "ndf", "delta_ndf_clean", "price_change_ndf_clean"),
]


def component_events(d, date_status):
    rows = []
    for _, r in d.iterrows():
        for label, prev, curr, delta, chg in components:
            if pd.notna(r[chg]) and float(r[chg]) == 1.0 and pd.notna(r[delta]):
                pct = np.nan
                if pd.notna(r[prev]) and float(r[prev]) != 0:
                    pct = 100.0 * float(r[delta]) / float(r[prev])
                rows.append({
                    "date_status": date_status,
                    "survey_year": int(r["survey_year"]),
                    "phone7": int(r["phone7"]),
                    "canonical_name": r["canonical_name"],
                    "component": label,
                    "previous_price": r[prev],
                    "current_price": r[curr],
                    "dollar_change": r[delta],
                    "absolute_dollar_change": abs(float(r[delta])),
                    "percent_change": pct,
                    "absolute_percent_change": abs(pct) if pd.notna(pct) else np.nan,
                })
    return pd.DataFrame(rows)

same = main[main["formal_revision_clean"].eq(0) & main["any_headline_change_clean"].eq(1)].copy()
new = main[main["formal_revision_clean"].eq(1) & main["any_headline_change_clean"].eq(1)].copy()
ce_same = component_events(same, "same_gpl_date")
ce_new = component_events(new, "new_gpl_date")
ce = pd.concat([ce_same, ce_new], ignore_index=True)
ce.to_csv(OUT / "phase3d_changed_component_events.csv", index=False)

pair_rows = []
for _, r in same.iterrows():
    dollars, pcts = [], []
    for _, prev, _, delta, chg in components:
        if pd.notna(r[chg]) and float(r[chg]) == 1.0 and pd.notna(r[delta]):
            dollars.append(abs(float(r[delta])))
            if pd.notna(r[prev]) and float(r[prev]) != 0:
                pcts.append(abs(100.0 * float(r[delta]) / float(r[prev])))
    pair_rows.append({
        "survey_year": int(r["survey_year"]),
        "phone7": int(r["phone7"]),
        "canonical_name": r["canonical_name"],
        "n_headline_components_changed": int(r["n_headline_changed_clean"]),
        "max_absolute_dollar_change": max(dollars),
        "mean_absolute_dollar_change": float(np.mean(dollars)),
        "max_absolute_percent_change": max(pcts) if pcts else np.nan,
        "mean_absolute_percent_change": float(np.mean(pcts)) if pcts else np.nan,
    })
pairmag = pd.DataFrame(pair_rows)
pairmag.to_csv(OUT / "phase3d_same_date_pair_magnitudes.csv", index=False)

mag_rows = [
    {"metric": "same_date_repricing_intervals", "estimate": len(same), "denominator": len(main[main["formal_revision_clean"].eq(0)])},
    {"metric": "same_date_changed_component_events", "estimate": len(ce_same), "denominator": np.nan},
    {"metric": "median_abs_dollar_change_same_date_component", "estimate": ce_same["absolute_dollar_change"].median(), "denominator": len(ce_same)},
    {"metric": "mean_abs_dollar_change_same_date_component", "estimate": ce_same["absolute_dollar_change"].mean(), "denominator": len(ce_same)},
    {"metric": "median_abs_pct_change_same_date_component", "estimate": ce_same["absolute_percent_change"].median(), "denominator": len(ce_same)},
    {"metric": "median_abs_pct_change_new_date_component", "estimate": ce_new["absolute_percent_change"].median(), "denominator": len(ce_new)},
    {"metric": "median_max_abs_dollar_change_same_date_interval", "estimate": pairmag["max_absolute_dollar_change"].median(), "denominator": len(pairmag)},
    {"metric": "share_same_date_intervals_max_change_ge_100", "estimate": pairmag["max_absolute_dollar_change"].ge(100).mean(), "denominator": len(pairmag)},
    {"metric": "share_same_date_intervals_max_pct_gt_5", "estimate": pairmag["max_absolute_percent_change"].gt(5).mean(), "denominator": len(pairmag)},
    {"metric": "share_same_date_intervals_two_or_more_components", "estimate": pairmag["n_headline_components_changed"].ge(2).mean(), "denominator": len(pairmag)},
]
pd.DataFrame(mag_rows).to_csv(OUT / "phase3d_magnitude_summary.csv", index=False)

# -----------------------------------------------------------------------------
# 2. Lead/lag timing audit
# -----------------------------------------------------------------------------
interval = df[df["gap_years_clean"].eq(1) & df["formal_revision_clean"].notna()].copy()
formal_map = {(int(r.phone7), int(r.survey_year)): float(r.formal_revision_clean) for _, r in interval.iterrows()}
main["formal_prev_interval"] = [formal_map.get((int(ph), int(y) - 1), np.nan) for ph, y in zip(main.phone7, main.survey_year)]
main["formal_next_interval"] = [formal_map.get((int(ph), int(y) + 1), np.nan) for ph, y in zip(main.phone7, main.survey_year)]

# General event-time rates around repricing/no-repricing intervals.
event_rows = []
for chg, label in [(1.0, "repricing"), (0.0, "no_repricing")]:
    z = main[main["any_headline_change_clean"].eq(chg)]
    for col, event_time in [("formal_prev_interval", -1), ("formal_revision_clean", 0), ("formal_next_interval", 1)]:
        q = z[col].dropna()
        event_rows.append({"price_status": label, "event_time": event_time, "n": len(q), "gpl_date_change_count": int(q.sum()), "gpl_date_change_rate": q.mean()})
pd.DataFrame(event_rows).to_csv(OUT / "phase3d_event_time_gpl_rates.csv", index=False)

# Most relevant comparison: among intervals with the same current GPL date, is
# repricing more likely to be adjacent to a GPL-date change than non-repricing?
sd = main[main["formal_revision_clean"].eq(0)].copy()
sd["any_adjacent_observed"] = sd[["formal_prev_interval", "formal_next_interval"]].notna().any(axis=1)
sd["any_adjacent_revision"] = sd[["formal_prev_interval", "formal_next_interval"]].fillna(0).max(axis=1)
timing_rows = []
for chg, label in [(1.0, "same_date_repricing"), (0.0, "same_date_no_repricing")]:
    z = sd[sd["any_headline_change_clean"].eq(chg)]
    for col, window in [("formal_prev_interval", "preceding_interval"), ("formal_next_interval", "following_interval")]:
        q = z[col].dropna()
        timing_rows.append({"group": label, "window": window, "n": len(q), "revision_count": int(q.sum()), "rate": q.mean()})
    zobs = z[z["any_adjacent_observed"]].copy()
    timing_rows.append({"group": label, "window": "either_adjacent_interval", "n": len(zobs), "revision_count": int(zobs["any_adjacent_revision"].sum()), "rate": zobs["any_adjacent_revision"].mean()})
pd.DataFrame(timing_rows).to_csv(OUT / "phase3d_adjacent_timing_summary.csv", index=False)

sdreg = sd[sd["any_adjacent_observed"]].copy()
m_adj = smf.ols("any_adjacent_revision ~ any_headline_change_clean + C(survey_year)", data=sdreg).fit(
    cov_type="cluster", cov_kwds={"groups": sdreg["phone7"]}
)
pd.DataFrame([{
    "outcome": "any_adjacent_gpl_date_change",
    "sample": "same_current_gpl_date_with_at_least_one_adjacent_interval_observed",
    "coefficient_on_repricing": m_adj.params["any_headline_change_clean"],
    "se_cluster_establishment": m_adj.bse["any_headline_change_clean"],
    "p": m_adj.pvalues["any_headline_change_clean"],
    "n": int(m_adj.nobs),
    "clusters": int(sdreg["phone7"].nunique()),
}]).to_csv(OUT / "phase3d_adjacent_timing_regression.csv", index=False)

# -----------------------------------------------------------------------------
# 3. Information / signal metrics
# -----------------------------------------------------------------------------
tp = int((main["formal_revision_clean"].eq(1) & main["any_headline_change_clean"].eq(1)).sum())
fp = int((main["formal_revision_clean"].eq(1) & main["any_headline_change_clean"].eq(0)).sum())
fn = int((main["formal_revision_clean"].eq(0) & main["any_headline_change_clean"].eq(1)).sum())
tn = int((main["formal_revision_clean"].eq(0) & main["any_headline_change_clean"].eq(0)).sum())
confusion = pd.DataFrame([
    {"gpl_date_signal": "new_date", "repricing": 1, "count": tp},
    {"gpl_date_signal": "new_date", "repricing": 0, "count": fp},
    {"gpl_date_signal": "same_date", "repricing": 1, "count": fn},
    {"gpl_date_signal": "same_date", "repricing": 0, "count": tn},
])
confusion.to_csv(OUT / "phase3d_signal_confusion_matrix.csv", index=False)
metric_defs = {
    "sensitivity": (tp, tp + fn),
    "specificity": (tn, tn + fp),
    "positive_predictive_value": (tp, tp + fp),
    "negative_predictive_value": (tn, tn + fn),
    "accuracy": (tp + tn, tp + tn + fp + fn),
}
sig_rows = []
for name, (num, den) in metric_defs.items():
    lo, hi = proportion_confint(num, den, method="wilson")
    sig_rows.append({"metric": name, "numerator": num, "denominator": den, "estimate": num / den, "wilson_lo": lo, "wilson_hi": hi})
pd.DataFrame(sig_rows).to_csv(OUT / "phase3d_signal_metrics.csv", index=False)

# -----------------------------------------------------------------------------
# 4. Wild-cluster bootstrap and leave-one-out inference
# -----------------------------------------------------------------------------
def cluster_cov(X, resid, clusters):
    n, k = X.shape
    uniq = np.unique(clusters)
    G = len(uniq)
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for g in uniq:
        ix = clusters == g
        score = X[ix].T @ resid[ix]
        meat += np.outer(score, score)
    factor = (G / (G - 1)) * ((n - 1) / (n - k))
    return factor * bread @ meat @ bread


def wild_cluster_bootstrap_p(full_formula, restricted_formula, data, label, B=B_WILD, seed=SEED):
    y, X = patsy.dmatrices(full_formula, data, return_type="dataframe")
    yv, Xv = np.asarray(y).ravel(), np.asarray(X)
    names = X.design_info.column_names
    idx = names.index("formal_revision_clean")
    clusters = data["phone7"].to_numpy()
    uniq = np.unique(clusters)
    clidx = [np.where(clusters == g)[0] for g in uniq]

    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    resid = yv - Xv @ beta
    V = cluster_cov(Xv, resid, clusters)
    se = math.sqrt(V[idx, idx])
    t_obs = beta[idx] / se

    _, Xr = patsy.dmatrices(restricted_formula, data, return_type="dataframe")
    Xrv = np.asarray(Xr)
    br = np.linalg.lstsq(Xrv, yv, rcond=None)[0]
    fit_r = Xrv @ br
    u_r = yv - fit_r

    n, k = Xv.shape
    G = len(uniq)
    inv = np.linalg.inv(Xv.T @ Xv)
    factor = (G / (G - 1)) * ((n - 1) / (n - k))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(B):
        weights = rng.choice(np.array([-1.0, 1.0]), size=G)
        ustar = np.empty_like(u_r)
        for j, ix in enumerate(clidx):
            ustar[ix] = u_r[ix] * weights[j]
        ystar = fit_r + ustar
        bstar = np.linalg.lstsq(Xv, ystar, rcond=None)[0]
        rstar = ystar - Xv @ bstar
        meat = np.zeros((k, k))
        for ix in clidx:
            score = Xv[ix].T @ rstar[ix]
            meat += np.outer(score, score)
        Vstar = factor * inv @ meat @ inv
        se_star = math.sqrt(max(Vstar[idx, idx], 1e-16))
        t_star = bstar[idx] / se_star
        exceed += abs(t_star) >= abs(t_obs)
    return {
        "specification": label,
        "coefficient": beta[idx],
        "cluster_se": se,
        "t_obs": t_obs,
        "wild_cluster_p": (1 + exceed) / (B + 1),
        "bootstrap_repetitions": B,
        "n": n,
        "clusters": G,
    }

wild = [
    wild_cluster_bootstrap_p(
        "any_headline_change_clean ~ formal_revision_clean + C(survey_year)",
        "any_headline_change_clean ~ C(survey_year)",
        main, "year_FE", B=B_WILD, seed=SEED,
    ),
    wild_cluster_bootstrap_p(
        "any_headline_change_clean ~ formal_revision_clean + prior_gpl_age_years + C(phone7) + C(survey_year)",
        "any_headline_change_clean ~ prior_gpl_age_years + C(phone7) + C(survey_year)",
        main, "establishment_and_year_FE_plus_prior_GPL_age", B=B_WILD, seed=SEED + 1,
    ),
]
pd.DataFrame(wild).to_csv(OUT / "phase3d_wild_cluster_bootstrap.csv", index=False)

loo_year = []
for yout in sorted(main["survey_year"].unique()):
    d = main[main["survey_year"].ne(yout)].copy()
    m = smf.ols("any_headline_change_clean ~ formal_revision_clean + prior_gpl_age_years + C(phone7) + C(survey_year)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["phone7"]}
    )
    loo_year.append({
        "omitted_year": int(yout), "n": int(m.nobs),
        "coefficient": m.params["formal_revision_clean"],
        "cluster_se": m.bse["formal_revision_clean"], "p": m.pvalues["formal_revision_clean"],
    })
pd.DataFrame(loo_year).to_csv(OUT / "phase3d_leave_one_year_out.csv", index=False)

sync = analysis.build_sync(df)

def sync_gap(d):
    mask = d["any_pair_active"].eq(1) & d["prior_price_vector_different"].eq(1)
    sb = d[mask & d["same_brand"]]
    cp = d[mask & (~d["same_brand"]) & (~d["same_owner"])]
    return {
        "n_same_brand": len(sb), "same_brand_rate": sb["exact_component_change_vector"].mean(),
        "n_comparator": len(cp), "comparator_rate": cp["exact_component_change_vector"].mean(),
        "gap": sb["exact_component_change_vector"].mean() - cp["exact_component_change_vector"].mean(),
    }

loo_brand_rows = []
base = sync_gap(sync)
loo_brand_rows.append({"omitted_brand": "NONE", **base, "permutation_p": np.nan, "permutations": np.nan})
brands = list(sync.loc[sync["same_brand"], "brand_a"].value_counts().index)
for brand in brands:
    d_sync = sync[sync["brand_a"].ne(brand) & sync["brand_b"].ne(brand)].copy()
    gap = sync_gap(d_sync)
    ppanel = df[df["brand_group"].ne(brand)].copy()
    bench, _ = analysis.permutation_benchmark_group(ppanel, "brand_group", nperm=B_PERM_LOO)
    brow = bench[bench["metric"].eq("active_baseline_diff_exact_component_vector")].iloc[0]
    loo_brand_rows.append({"omitted_brand": brand, **gap, "permutation_p": brow["one_sided_p"], "permutations": B_PERM_LOO})
pd.DataFrame(loo_brand_rows).to_csv(OUT / "phase3d_leave_one_brand_out.csv", index=False)

print("Phase 3D complete")
print(f"Main N={len(main)}, same-date repricing N={len(same)}")
print(pd.read_csv(OUT / "phase3d_magnitude_summary.csv").to_string(index=False))
print(pd.read_csv(OUT / "phase3d_signal_metrics.csv").to_string(index=False))
print(pd.read_csv(OUT / "phase3d_adjacent_timing_summary.csv").to_string(index=False))
print(pd.read_csv(OUT / "phase3d_wild_cluster_bootstrap.csv").to_string(index=False))
print(pd.read_csv(OUT / "phase3d_leave_one_brand_out.csv").to_string(index=False))
