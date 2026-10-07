#!/usr/bin/env python3
"""Kansas City measurement validation for the GPL-date paper.

No Kansas City records are joined to Greater Austin. Input is an explicitly
source-linked, visually reviewed transcription, not automatic OCR output.
FCA-GKC has authorized publication and redistribution of its survey data. The
identified input (kc_later_audited_records.csv) is read from the path given by --input.
All coefficients below are descriptive differences in conditional sample means.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

COMPONENTS = ("direct_cremation", "immediate_burial", "standard_funeral")
ACCEPTED = {"anchor", "same_phone_address_and_provider_family"}
WINDOWS = ((2018, 2020), (2020, 2023))
THRESHOLDS = (("gt5", 5, None), ("gt25", 25, None),
              ("gt2p5pct", None, .025), ("gt5pct", None, .05))

def read_panel(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    seen = set()
    for r in rows:
        for c in ("source_wave", "branch_reference_id"):
            r[c] = int(r[c])
        key = (r["source_wave"], r["branch_reference_id"])
        if key in seen:
            raise ValueError(f"Duplicate branch-wave key: {key}")
        seen.add(key)
        if r["date_precision"] != "month":
            raise ValueError("Later-wave KC dates must retain their published month precision.")
        r["carried_forward"] = r["carried_forward"] == "True"
        for c in COMPONENTS:
            r[c] = float(r[c]) if r[c] else None
            if r[c] is not None and r[c] <= 0:
                raise ValueError("An observed price must be positive.")
    return rows

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def build_intervals(rows: list[dict]) -> list[dict]:
    by = {y: {r["branch_reference_id"]: r for r in rows
              if r["source_wave"] == y} for y in (2018, 2020, 2023)}
    pairs = []
    for y0, y1 in WINDOWS:
        for key in sorted(by[y0].keys() & by[y1].keys()):
            a, b = by[y0][key], by[y1][key]
            date_ok = bool(a["gpl_month"] and b["gpl_month"])
            row = {
                "branch_reference_id": key, "market": "Greater Kansas City",
                "wave0": y0, "wave1": y1, "gap_years": y1 - y0,
                "strict_link": a["linkage_status"] in ACCEPTED and b["linkage_status"] in ACCEPTED,
                "link_status0": a["linkage_status"], "link_status1": b["linkage_status"],
                "gpl_month0": a["gpl_month"], "gpl_month1": b["gpl_month"],
                "dates_observed": date_ok,
                "changed_gpl_month": int(a["gpl_month"] != b["gpl_month"]) if date_ok else None,
                "backward_gpl_month": int(b["gpl_month"] < a["gpl_month"]) if date_ok else None,
                "carry0": a["carried_forward"], "carry1": b["carried_forward"],
                "carry_either": a["carried_forward"] or b["carried_forward"],
                "known_source_wave0": 2018 if a["carried_forward"] else None,
                "known_source_wave1": 2018 if b["carried_forward"] else None,
                "source_vintage_note": "An unflagged record is not proof of a newly obtained GPL.",
            }
            for c in COMPONENTS:
                row[c+"_p0"], row[c+"_p1"] = a[c], b[c]
                row[c+"_observed"] = a[c] is not None and b[c] is not None
                row[c+"_comparable"] = c != "immediate_burial" or y0 >= 2020
                row[c+"_known_carry_excluded"] = row["carry_either"]
            pairs.append(row)
    if any(p["gap_years"] == 1 for p in pairs):
        raise AssertionError("KC irregular intervals must not become annual observations.")
    return pairs

def select(pairs, y0, y1, component, strict=True, exclude_carry=True):
    return [p for p in pairs if (p["wave0"], p["wave1"]) == (y0, y1)
            and (p["strict_link"] or not strict) and p["dates_observed"]
            and p[component+"_observed"] and p[component+"_comparable"]
            and (not p["carry_either"] or not exclude_carry)]

def changed(p, c, dollars, pct):
    x, y = p[c+"_p0"], p[c+"_p1"]
    return int(abs(y-x) > dollars) if dollars is not None else int(abs(y-x)/x > pct)

def summary(sample, y0, y1, c, threshold, dollars, pct, scenario):
    d1 = [p for p in sample if p["changed_gpl_month"] == 1]
    d0 = [p for p in sample if p["changed_gpl_month"] == 0]
    n1,n0=len(d1),len(d0)
    k1,k0=sum(changed(p,c,dollars,pct) for p in d1),sum(changed(p,c,dollars,pct) for p in d0)
    return {"market":"Greater Kansas City","window":f"{y0}-{y1}","gap_years":y1-y0,
            "outcome":c,"threshold":threshold,"scenario":scenario,"n":len(sample),
            "changed_month_n":n1,"changed_month_reprices":k1,
            "unchanged_month_n":n0,"unchanged_month_reprices":k0,
            "changed_month_rate":k1/n1 if n1 else None,
            "unchanged_month_rate":k0/n0 if n0 else None,
            "descriptive_LPM_coefficient":k1/n1-k0/n0 if n1 and n0 else None,
            "date_precision":"month; equality does not establish exact-day equality",
            "inference":"descriptive only; common pricing processes and small same-month cells",
            "definition_qualification":("2018 crematory-fee inclusion not explicitly established; published-price diagnostic"
                 if y0==2018 and c=="direct_cremation" else
                 "published target-definition comparison; not original-GPL reconstruction")}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True,help="Restricted, source-linked KC transcription CSV")
    ap.add_argument("--out",type=Path,required=True,help="Aggregate output folder (no identifying rows)")
    ap.add_argument("--private-out",type=Path,default=None,help="Optional restricted interval ledger folder")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=read_panel(args.input)
    pairs=build_intervals(rows)
    counts={y:sum(r["source_wave"]==y for r in rows) for y in (2018,2020,2023)}
    if counts != {2018:111,2020:113,2023:106}:
        raise AssertionError(f"Unexpected transcribed anchor-universe counts: {counts}")
    r20=[r for r in rows if r["source_wave"]==2020]
    carries=[r for r in r20 if r["carried_forward"]]
    old={r["branch_reference_id"]:r for r in rows if r["source_wave"]==2018}
    carry_mismatches=sum(any(r[c] != old[r["branch_reference_id"]][c] for c in COMPONENTS) for r in carries)
    if len(carries)!=25 or carry_mismatches:
        raise AssertionError("Carry-forward audit changed; return to the source pages.")
    flows=[]; summaries=[]
    for y0,y1 in WINDOWS:
        cand=[p for p in pairs if (p["wave0"],p["wave1"])==(y0,y1)]
        strict=[p for p in cand if p["strict_link"]]
        dated=[p for p in strict if p["dates_observed"]]
        priced=[p for p in dated if p["direct_cremation_observed"]]
        filtered=[p for p in priced if not p["carry_either"]]
        flows.append({"window":f"{y0}-{y1}","gap_years":y1-y0,
            "candidate_links":len(cand),"strict_branch_links":len(strict),
            "dated_branch_links":len(dated),"dated_DC_links":len(priced),
            "explicit_carry_exclusions":len(priced)-len(filtered),
            "usable_DC_links":len(filtered)})
        for c in COMPONENTS:
            sample=select(pairs,y0,y1,c)
            if not sample: continue
            for label,dol,pct in THRESHOLDS:
                summaries.append(summary(sample,y0,y1,c,label,dol,pct,"strict_no_known_carry"))
            # Looser identity is a sensitivity, not the headline sample.
            summaries.append(summary(select(pairs,y0,y1,c,strict=False),y0,y1,c,"gt5",5,None,
                                     "candidate_identity_sensitivity_no_known_carry"))
        # Deliberately naive comparison is for explaining copied records, not estimating true behavior.
        summaries.append(summary(select(pairs,y0,y1,"direct_cremation",exclude_carry=False),
                                 y0,y1,"direct_cremation","gt5",5,None,"diagnostic_includes_known_copies_NOT_current_prices"))
    write_csv(args.out/"kc_interval_flow.csv",flows)
    write_csv(args.out/"kc_market_specific_associations.csv",summaries)
    dc=[s for s in summaries if s["outcome"]=="direct_cremation"
        and s["threshold"]=="gt5" and s["scenario"]=="strict_no_known_carry"]
    if [(s["n"],s["changed_month_n"],s["changed_month_reprices"],s["unchanged_month_n"],s["unchanged_month_reprices"]) for s in dc] != [(81,59,37,22,1),(75,68,54,7,1)]:
        raise AssertionError("The reviewed conditional-frequency table was not reproduced.")
    write_csv(args.out/"kc_direct_cremation_validation.csv",dc)
    audit={"source_2020_rows":113,"explicit_carry_forward_rows":25,
           "carry_forward_share":25/113,"copied_vectors_equal_2018":25,
           "missing_gpl_month_among_carries":sum(not r["gpl_month"] for r in carries),
           "input_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
           "2009_2012_date_test":"NOT_IDENTIFIED: supplied 2009 publication lacks establishment GPL dates",
           "2025_workbook":"Unpublished collection frame; no prices; not used in a price interval",
           "irregular_gaps":[2,3],"joined_to_Austin":False,
           "source_scope":"published survey records, not provider-original GPL histories"}
    (args.out/"kc_validation_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    # A compact, standalone table; no provider identities.
    ls=[r"\begin{table}[htbp]\centering",r"\caption{Kansas City: recorded date changes and direct-cremation prices}\label{tab:kc_dates}",
        r"\begin{threeparttable}\small",r"\begin{tabular}{lrrrr}\toprule",
        r"Survey interval & $N$ & Changed month & Same month & Gap (pp)\\\midrule"]
    for s in dc:
        ls.append(f'{s["window"].replace("-", "--")} & {s["n"]} & '
                  f'{s["changed_month_reprices"]}/{s["changed_month_n"]} ({100*s["changed_month_rate"]:.1f}\\%) & '
                  f'{s["unchanged_month_reprices"]}/{s["unchanged_month_n"]} ({100*s["unchanged_month_rate"]:.1f}\\%) & '
                  f'{100*s["descriptive_LPM_coefficient"]:.1f}\\\\')
    ls += [r"\bottomrule\end{tabular}",r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item Notes: Change means an absolute difference exceeding \$5. Dates are published month--year fields, not exact effective dates. Linked physical branches require agreement on phone, address and provider family; unresolved identity or online-only scope is excluded. Known 2018 prices copied into the 2020 publication are excluded at either endpoint. An unflagged record is not independently verified as newly collected. The 2018--2020 row is a qualified published-price diagnostic because 2018 crematory-fee inclusion is not explicit. The 2020 and 2023 texts target inclusion but advise provider verification. Gaps are differences in sample proportions, not causal effects; same-month cells are small and branches may share pricing processes. These are two- and three-year intervals, never annual observations.",
        r"\item Source: Funeral Consumers Alliance of Greater Kansas City, published surveys for 2018, 2020 and 2023; author source audit and calculations.",
        r"\end{tablenotes}\end{threeparttable}\end{table}"]
    (args.out/"tableA10_kc_dates.tex").write_text("\n".join(ls)+"\n")
    if args.private_out:
        args.private_out.mkdir(parents=True,exist_ok=True)
        write_csv(args.private_out/"kc_linked_intervals_RESTRICTED.csv",pairs)
    print(json.dumps({"status":"PASS","rows":counts,"flows":flows,
                      "headline_counts":[{k:s[k] for k in ("window","n","changed_month_reprices","changed_month_n","unchanged_month_reprices","unchanged_month_n")} for s in dc]},indent=2))

if __name__=="__main__":
    main()
