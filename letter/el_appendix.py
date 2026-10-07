#!/usr/bin/env python3
"""Referee-facing diagnostics for the online appendix (and several main-text numbers).
Uses only audited headline prices (direct cremation, immediate burial, basic-services fee).
Embalming and the service composite are NOT used: the panel's embalming column has not
been audited against the survey pages, and the composite changes definition in 2017. Writes output/appendix.json and manuscript/tables/appA*.tex."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
REP = Path(os.environ.get('AUSTIN_PIPELINE', '../austin_pipeline')); OUT = Path('output'); TAB = Path('manuscript/tables')
src = open('el_analysis.py').read(); ns = {}
exec(src[:src.index("res = {}")], ns)
z1, m, build, d = ns['z1'], ns['m'], ns['build'], ns['d']
HEAD = ['direct_cremation', 'immediate_burial', 'ndf']
res = {}

def sm(D, R):
    ok = pd.notna(R) & pd.notna(D); D = np.asarray(D)[ok]; R = np.asarray(R)[ok]
    tp = int(((D == 1) & (R == 1)).sum()); fn = int(((D == 0) & (R == 1)).sum())
    fp = int(((D == 1) & (R == 0)).sum()); tn = int(((D == 0) & (R == 0)).sum())
    se = tp / (tp + fn); sp = tn / (tn + fp)
    return dict(n=tp + fn + fp + tn, new_n=tp + fp, new_chg=tp, same_n=fn + tn, same_chg=fn,
                rate_new=tp / (tp + fp), rate_same=fn / (fn + tn), sens=se, spec=sp, J=se + sp - 1)

# ---- A1: by price measure, annual and two-year; Kansas City for comparison
A1 = {}
for c, lab in [('ndf', 'Basic-services fee'), ('direct_cremation', 'Direct cremation'), ('immediate_burial', 'Immediate burial incl. casket')]:
    s = z1[z1.D.notna() & z1['chg_' + c].notna()]
    A1[f'{lab}, annual'] = sm(s.D, s['chg_' + c])
A1['Any headline price (main sample), annual'] = sm(m.D, m.R)
z2 = build(d, 2)
s2 = z2[z2.D.notna() & z2.chg_direct_cremation.notna()]
A1['Direct cremation, two-year links'] = sm(s2.D, s2.chg_direct_cremation)
m2 = z2[z2.D.notna() & z2.R.notna() & z2.n_common.ge(2)]
A1['Any headline price, two-year links'] = sm(m2.D, m2.R)
kc = pd.read_csv(OUT / 'kc_signal_metrics.csv')
for _, k in kc[kc.outcome.eq('direct_cremation')].iterrows():
    A1[f'Kansas City direct cremation, {k.window} (month dates)'] = dict(n=int(k.n), new_n=int(k.new_n), new_chg=int(k.new_reprice),
        same_n=int(k.same_n), same_chg=int(k.same_reprice), rate_new=k.rate_new, rate_same=k.rate_same, sens=k.sensitivity, spec=k.specificity, J=k.J)
res['A1_by_measure'] = A1

# ---- A2: does same-date repricing look like survey noise?
ev = pd.read_csv(REP / 'phase3d_output/phase3d_changed_component_events.csv')
cmap = {'direct_cremation': 'direct_cremation', 'immediate_burial': 'immediate_burial', 'basic_services_fee': 'ndf'}
nx = d.set_index(['phone7', 'survey_year'])
def nextp(r):
    k = (r.phone7, r.survey_year + 1)
    return nx.loc[k, cmap[r.component]] if k in nx.index else np.nan
ev['next_price'] = [nextp(r) for r in ev.itertuples()]
ev['reverts'] = np.where(ev.next_price.notna(), (ev.next_price - ev.previous_price).abs().le(5), np.nan)
A2 = {}
for st, g in ev.groupby('date_status'):
    lab = 'same' if st == 'same_gpl_date' else 'new'
    A2[lab] = dict(events=len(g), decreases=int((g.dollar_change < 0).sum()), share_decreases=float((g.dollar_change < 0).mean()),
                   next_year_observed=int(g.reverts.notna().sum()), reverts=int(np.nansum(g.reverts)),
                   revert_rate=float(np.nanmean(g.reverts)),
                   by_component={c: dict(events=int(len(h)), decreases=int((h.dollar_change < 0).sum()),
                                         observed=int(h.reverts.notna().sum()), reverts=int(np.nansum(h.reverts)))
                                 for c, h in g.groupby('component')})
A2['fisher_p_decreases'] = float(fisher_exact([[A2['same']['decreases'], A2['same']['events'] - A2['same']['decreases']],
                                               [A2['new']['decreases'], A2['new']['events'] - A2['new']['decreases']]])[1])
A2['fisher_p_reverts'] = float(fisher_exact([[A2['same']['reverts'], A2['same']['next_year_observed'] - A2['same']['reverts']],
                                             [A2['new']['reverts'], A2['new']['next_year_observed'] - A2['new']['reverts']]])[1])
# Persistent-change outcome: drop any component change that reverts to its t-1 price at t+1
nxt = d[['phone7', 'survey_year'] + HEAD].copy(); nxt['survey_year'] -= 1
mm = m.merge(nxt.rename(columns={c: 'N_' + c for c in HEAD}), on=['phone7', 'survey_year'], how='left')
keep, unk = [], []
for c in HEAD:
    comp = mm[c].notna() & mm['L_' + c].notna()
    chg = comp & (mm[c] - mm['L_' + c]).abs().gt(5)
    rev = chg & mm['N_' + c].notna() & (mm['N_' + c] - mm['L_' + c]).abs().le(5)
    keep.append(chg & ~rev); unk.append(chg & mm['N_' + c].isna())
mm['R_persist'] = pd.concat(keep, axis=1).any(axis=1).astype(float)
U = pd.concat(unk, axis=1).any(axis=1)
A2['persistent_keep_unverifiable'] = sm(mm.D, mm.R_persist)
A2['persistent_drop_unverifiable'] = sm(mm.D[~U], mm.R_persist[~U])
sd = m[(m.D == 0) & (m.R == 1)]
A2['same_date_repricing_establishments'] = int(sd.phone7.nunique())
A2['same_date_repricing_max_share_one_establishment'] = float(sd.phone7.value_counts().iloc[0] / len(sd))
res['A2_noise'] = A2

# ---- A3: robustness of the gap
rob = pd.read_csv(REP / 'adversarial_output/measurement_robustness.csv')
thr = pd.read_csv(REP / 'adversarial_output/headline_thresholds.csv')
A3 = []
def addrow(label, s):
    a, b = s[s.gpl_date_change.eq(1) if 'gpl_date_change' in s else s.date_change.eq(1)].iloc[0], s[s.gpl_date_change.eq(0) if 'gpl_date_change' in s else s.date_change.eq(0)].iloc[0]
    A3.append(dict(check=label, new=f"{int(a.repricing_count)}/{int(a.n)}", rate_new=float(a.rate), same=f"{int(b.repricing_count)}/{int(b.n)}", rate_same=float(b.rate), gap=float(a.rate - b.rate)))
for key, label in [('baseline_three_headline_prices_at_least_two_observed', 'Baseline'),
                   ('all_three_prices_observed_at_both_endpoints', 'All three prices observed'),
                   ('exclude_minimum_casket_burial_measure', 'Cremation or fee only (burial dropped)'),
                   ('both_endpoint_surveys_2019_or_later', 'Both surveys 2019 or later'),
                   ('omit_2016_to_2017_transition', 'Omit 2016--2017 survey redesign'),
                   ('exclude_flagged_website_intervals_and_2017_phone_clarification', 'Omit website/phone-sourced flags'),
                   ('exclude_entire_Gipson_and_Eeds_histories', 'Omit two audited histories')]:
    addrow(label, rob[rob['sample'].eq(key)])
for key, label in [('More than $25', 'Threshold \\$25'), ('More than 2.5 percent', 'Threshold 2.5\\%'), ('More than 5 percent', 'Threshold 5\\%')]:
    addrow(label, thr[thr.threshold.eq(key)])
P = A2['persistent_keep_unverifiable']
A3.append(dict(check='Drop changes reversed next year', new=f"{P['new_chg']}/{P['new_n']}", rate_new=P['rate_new'],
               same=f"{P['same_chg']}/{P['same_n']}", rate_same=P['rate_same'], gap=P['rate_new'] - P['rate_same']))
brand = d.dropna(subset=['brand_group']).groupby('phone7').brand_group.first()
mb = m.assign(brand=m.phone7.map(brand))
lobo = [float(q[q.D == 1].R.mean() - q[q.D == 0].R.mean()) for b in mb.brand.dropna().unique() for q in [mb[mb.brand.ne(b) | mb.brand.isna()]]]
loeo = [float(m[m.phone7.ne(u) & m.D.eq(1)].R.mean() - m[m.phone7.ne(u) & m.D.eq(0)].R.mean()) for u in m.phone7.unique()]
res['A3_robustness'] = A3
res['leave_one_brand_out_gap_range'] = [min(lobo), max(lobo)]
res['leave_one_establishment_out_gap_range'] = [min(loeo), max(loeo)]
bnd = pd.read_csv(REP / 'revision_output/attrition_gap_bounds.csv')
res['worst_case_gap'] = float(bnd.lower_bound_gap.iloc[0])
per = {}
for lab, q in [('2014--2019', m.survey_year.le(2019)), ('2020--2024', m.survey_year.ge(2020))]:
    per[lab] = sm(m[q].D, m[q].R)
res['by_period'] = per
json.dump(res, open(OUT / 'appendix.json', 'w'), indent=2, default=float)

# ---- LaTeX appendix tables
def f3(x): return f"{x:.3f}"
def pc(x): return f"{100*x:.1f}"
SHORT = {'Kansas City direct cremation, 2018-2020 (month dates)': 'Kansas City, cremation, 2018--20',
         'Kansas City direct cremation, 2020-2023 (month dates)': 'Kansas City, cremation, 2020--23',
         'Immediate burial incl. casket, annual': 'Burial incl.~casket, annual',
         'Any headline price (main sample), annual': 'Any of three (main sample), annual',
         'Any headline price, two-year links': 'Any of three, two-year links',
         'Basic-services fee, annual': 'Basic-services fee, annual'}
rows = []
for lab, r in A1.items():
    rows.append(f"{SHORT.get(lab, lab)} & {r['n']} & {r['new_chg']}/{r['new_n']} ({pc(r['rate_new'])}) & {r['same_chg']}/{r['same_n']} ({pc(r['rate_same'])}) & {f3(r['sens'])} & {f3(r['spec'])} & {f3(r['J'])} \\\\")
(TAB / 'appA1_measures.tex').write_text(r"""\begin{table}[H]\centering\footnotesize
\caption{Date signal by price measure and observation gap.}\label{tab:A1}
\setlength{\tabcolsep}{2.5pt}\scriptsize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{4.6cm}cccccc@{}}\toprule
Price measure, links & $N$ & New date & Same date & Sens. & Spec. & $J$ \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular*}
\begin{minipage}{\textwidth}\vspace{3pt}\scriptsize Notes: Repriced/links (\%) for new-date and same-date links. Single-measure rows use every link at which that price is comparable, so samples differ from the main sample. Two-year links compare survey $t-2$ with $t$. Kansas City rows use month-level dates from published FCA-GKC survey tables and exclude records marked as carried forward.\end{minipage}
\end{table}
""")
a, b = A2['same'], A2['new']
def comp_line(lab, key):
    x, y = a['by_component'].get(key, dict(events=0, decreases=0, observed=0, reverts=0)), b['by_component'][key]
    return f"{lab} & {x['events']} & {x['decreases']} & {x['reverts']}/{x['observed']} & {y['events']} & {y['decreases']} & {y['reverts']}/{y['observed']} \\\\"
(TAB / 'appA2_noise.tex').write_text(r"""\begin{table}[H]\centering\footnotesize
\caption{Do same-date price changes look like survey noise?}\label{tab:A2}
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{3.6cm}cccccc@{}}\toprule
 & \multicolumn{3}{c}{Same date} & \multicolumn{3}{c}{New date}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Component & Changes & Decreases & Reversed & Changes & Decreases & Reversed \\\midrule
""" + '\n'.join([comp_line('Direct cremation', 'direct_cremation'), comp_line('Burial incl.~casket', 'immediate_burial'),
                 comp_line('Basic-services fee', 'basic_services_fee')]) +
f"\nAll & {a['events']} & {a['decreases']} & {a['reverts']}/{a['next_year_observed']} & {b['events']} & {b['decreases']} & {b['reverts']}/{b['next_year_observed']} \\\\" + r"""
\bottomrule\end{tabular*}
\begin{minipage}{\textwidth}\vspace{3pt}\scriptsize Notes: Changed component events (more than \$5) in the 348 annual intervals. Reversed: the price at $t+1$ is within \$5 of its value at $t-1$, among events whose next-year price is observed. Fisher exact $p$-values for same- versus new-date differences: decreases """ + (r"$<0.001$" if A2['fisher_p_decreases'] < 0.001 else f"{A2['fisher_p_decreases']:.3f}") + ", reversals " + f"{A2['fisher_p_reverts']:.3f}" + r""".\end{minipage}
\end{table}
""")
rows = [f"{r['check']} & {r['new']} ({pc(r['rate_new'])}) & {r['same']} ({pc(r['rate_same'])}) & {r['gap']:.3f} \\\\" for r in A3]
rows.append(f"Worst-case missing price outcomes & & & {res['worst_case_gap']:.3f} \\\\")
rows.append(f"Leave one coded brand out (range) & & & {res['leave_one_brand_out_gap_range'][0]:.3f}--{res['leave_one_brand_out_gap_range'][1]:.3f} \\\\")
rows.append(f"Leave one establishment out (range) & & & {res['leave_one_establishment_out_gap_range'][0]:.3f}--{res['leave_one_establishment_out_gap_range'][1]:.3f} \\\\")
(TAB / 'appA3_robust.tex').write_text(r"""\begin{table}[H]\centering\footnotesize
\caption{Robustness of the same-date versus new-date repricing gap, Greater Austin.}\label{tab:A3}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{6.2cm}ccc@{}}\toprule
Check & New date & Same date & Gap \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular*}
\begin{minipage}{\textwidth}\vspace{3pt}\scriptsize Notes: Repriced/intervals (\%). Rows other than the last four and the reversal row reproduce the replication package's robustness outputs. Worst case: every missing new-date outcome set to no repricing and every missing same-date outcome to repricing. ``Drop changes reversed next year'' keeps changes whose next-year price is unobserved.\end{minipage}
\end{table}
""")
print(json.dumps({k: res[k] for k in ['A2_noise', 'leave_one_brand_out_gap_range', 'leave_one_establishment_out_gap_range', 'by_period']}, indent=1, default=float)[:3000])

# ---- A4: list-age regressions, extended; A5: simulation
age = pd.read_csv(OUT / 'list_age_tests.csv').set_index('model')
dif = pd.read_csv(OUT / 'proxy_difference_tests.csv').set_index('model')
sb = json.load(open(OUT / 'sim_boot.json'))
def cell(c, s, p): return f"{c:.3f}".replace('-', '$-$') + f" ({s:.3f}) [{p:.3f}]"
spec_rows = [
 ('Price change, same-date intervals only', '(5) price change on age, same-date'),
 ('Price change, new-date intervals only', '(6) price change on age, new-date'),
 ('Date change, cremation/fee sample', '(1nb) date change on age, cremation/fee sample'),
 ('Cremation or fee change, cremation/fee sample', '(2nb) cremation/fee change on age (both comparable)'),
]
rows = [f"{lab} & {cell(age.loc[k,'coef'], age.loc[k,'se'], age.loc[k,'wild_p'])} & {int(age.loc[k,'n'])} \\\\" for lab, k in spec_rows]
for lab, k in [('Date minus price change, establishment and year effects', 'D minus R on age, unit+year FE'),
               ('Date minus price change, year effects only', 'D minus R on age, year FE')]:
    rows.append(f"{lab} & {cell(dif.loc[k,'coef'], dif.loc[k,'se'], dif.loc[k,'wild_p'])} & {int(dif.loc[k,'n'])} \\\\")
(TAB / 'appA4_age.tex').write_text(r"""\begin{table}[H]\centering\footnotesize
\caption{Prior list age: additional specifications.}\label{tab:A4}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{7cm}cc@{}}\toprule
Outcome and sample & Coefficient on prior list age & $N$ \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular*}
\begin{minipage}{\textwidth}\vspace{3pt}\scriptsize Notes: Linear probability models. The first five rows include establishment and year effects. Establishment-clustered standard errors in parentheses; wild-cluster bootstrap $p$-values (4,999 Rademacher draws) in brackets. The cremation/fee sample is the 304 intervals at which both direct cremation and the basic-services fee are comparable. ``Date minus price'' regresses $D-R$ on age and tests whether the two gradients are equal.\end{minipage}
\end{table}
""")
sims = [('Homogeneous hazard, $p=0.65$', 'sim_homogeneous_p_0.65'), ('Heterogeneous, Beta(2.6, 1.4)', 'sim_heterogeneous_beta_2.6_1.4'),
        ('More heterogeneous, Beta(1.3, 0.7)', 'sim_more_dispersed_beta_1.3_0.7')]
rows = [f"{lab} & {sb[k]['unit_year_FE_mean']:.3f} ({sb[k]['unit_year_FE_sd']:.3f}) & {sb[k]['year_FE_mean']:.3f} ({sb[k]['year_FE_sd']:.3f}) \\\\".replace('-0', '$-$0') for lab, k in sims]
rows.append(f"Austin estimate, Table 2 column (1) & {age.loc['(1) date change on prior age','coef']:.3f} & {age.loc['(1y) date change on prior age, year FE','coef']:.3f} \\\\".replace('-0', '$-$0'))
(TAB / 'appA5_sim.tex').write_text(r"""\begin{table}[H]\centering\footnotesize
\caption{List-age coefficients when the true hazard is constant.}\label{tab:A5}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcc@{}}\toprule
Simulated process & Establishment and year effects & Year effects only \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular*}
\begin{minipage}{\textwidth}\vspace{3pt}\scriptsize Notes: Mean (standard deviation) of the coefficient on prior list age in a regression of the date-change indicator, over 500 simulated panels of 53 providers and seven annual intervals. Each provider re-dates with a constant probability drawn as shown (mean 0.65, matching Austin's 64.9\% date-change frequency), so there is no duration dependence. Initial list age is uniform on $[0,3]$ years; a list issued during an interval is uniform on $[0,1]$ years old at the next survey; otherwise age rises by one year.\end{minipage}
\end{table}
""")
print('appendix tables written')
