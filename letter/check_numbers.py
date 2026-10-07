#!/usr/bin/env python3
"""Check every number stated in the manuscript text against the computed outputs.
Each row: (claim label, literal as printed in manuscript.tex, computed value formatted, source).
Fails loudly if a literal is missing from the text or disagrees with the computation.
Also writes output/number_checks.csv, which doubles as the re-verification list."""
import json, re, sys
import pandas as pd
from pathlib import Path
R = Path(__import__('os').environ.get('AUSTIN_PIPELINE','../austin_pipeline'))
tex = Path('manuscript/manuscript.tex').read_text()
tex_nb = tex
r = json.load(open('output/el_results.json')); sb = json.load(open('output/sim_boot.json')); ap = json.load(open('output/appendix.json'))
age = pd.read_csv('output/list_age_tests.csv').set_index('model')
kc = pd.read_csv('output/kc_signal_metrics.csv'); kc = kc[kc.outcome.eq('direct_cremation')].set_index('window')
mag = pd.read_csv(R/'phase3d_output/phase3d_magnitude_summary.csv').set_index('metric').estimate
rob = pd.read_csv(R/'adversarial_output/measurement_robustness.csv')
thr = pd.read_csv(R/'adversarial_output/headline_thresholds.csv')
bnd = pd.read_csv(R/'revision_output/attrition_gap_bounds.csv')
wild = pd.read_csv(R/'adversarial_output/models_and_cluster_inference.csv')
d = pd.read_csv(R/'analytic_panel.csv'); d13 = d[d.survey_year <= 2024]
T = json.load(open('output/panel_T.json'))
def p1(x): return f'{100*x:.1f}'
def f3(x): return f'{x:.3f}'
def rr(s, v): x = rob[rob['sample'].eq(s) & rob.gpl_date_change.eq(v)].rate.iloc[0]; return p1(x)
A = lambda k, c='coef': age.loc[k, c]
two = r['two_year']['all_two_year_links']
C = [
 ('intervals', '348 audited annual intervals', f"{r['n']} audited annual intervals", 'el_results n'),
 ('new-date repricing rate', '87.2\\%', p1(r['rate_new'])+'\\%', 'el_results rate_new'),
 ('same-date repricing rate', '32.8\\%', p1(r['rate_same'])+'\\%', 'el_results rate_same'),
 ('sensitivity pct (abstract)', '83.1\\%', p1(r['sensitivity'])+'\\%', 'el_results sensitivity'),
 ('specificity pct (abstract)', '73.9\\%', p1(r['specificity'])+'\\%', 'el_results specificity'),
 ('J pct', '57\\%', f"{round(100*r['J'])}\\%", 'el_results J'),
 ('date freq', '64.9\\%', p1(r['freq_date'])+'\\%', 'el_results freq_date'),
 ('price freq', '68.1\\%', p1(r['freq_price'])+'\\%', 'el_results freq_price'),
 ('J', '0.570', f3(r['J']), 'el_results J'),
 ('records 2013-2024', '622 establishment-year records for 71', f"{len(d13)} establishment-year records for {d13.phone7.nunique()}", 'analytic_panel.csv, survey_year<=2024'),
 ('establishments in sample', '348 intervals from 53 establishments', f"{r['n']} intervals from {r['units']} establishments", 'el_results units'),
 ('new-date count', '197 of 226', f"{r['new_date_reprice']} of {r['new_date_n']}", '2x2'),
 ('same-date count', '40 of 122', f"{r['same_date_reprice']} of {r['same_date_n']}", '2x2'),
 ('sensitivity', 'sensitivity 0.831', 'sensitivity '+f3(r['sensitivity']), '2x2'),
 ('sens numerator', '197 of 237', f"{r['new_date_reprice']} of {r['new_date_reprice']+r['same_date_reprice']}", '2x2'),
 ('specificity', 'specificity 0.739', 'specificity '+f3(r['specificity']), '2x2'),
 ('spec numerator', '82 of 111', f"{r['same_date_n']-r['same_date_reprice']} of {r['same_date_n']-r['same_date_reprice']+r['new_date_n']-r['new_date_reprice']}", '2x2'),
 ('coef year FE', '0.529 (0.063)', f"{A('headline: year FE'):.3f} ({A('headline: year FE','se'):.3f})", 'list_age_tests'),
 ('coef unit+year FE', '0.557 (0.075)', f"{A('headline: unit+year FE'):.3f} ({A('headline: unit+year FE','se'):.3f})", 'list_age_tests'),
 ('coef + age', '0.559 (0.076)', f"{A('headline: unit+year FE + prior age'):.3f} ({A('headline: unit+year FE + prior age','se'):.3f})", 'list_age_tests'),
 ('wild p < 0.001', 'below 0.001', 'below 0.001' if wild.wild_p.max() < 0.001 else 'FAIL', 'models_and_cluster_inference wild_p'),
 ('same-date component events', '49 changed prices', f"{int(mag['same_date_changed_component_events'])} changed prices", 'phase3d_magnitude_summary'),
 ('median $ change', '\\$150 (7.3\\%)', f"\\${mag['median_abs_dollar_change_same_date_component']:.0f} ({mag['median_abs_pct_change_same_date_component']:.1f}\\%)", 'phase3d_magnitude_summary'),
 ('>= $100', '32 of the 40', f"{round(mag['share_same_date_intervals_max_change_ge_100']*40)} of the 40", 'phase3d share_same_date_intervals_max_change_ge_100'),
 ('all three prices', '(87.4\\% versus 34.0\\%)', f"({rr('all_three_prices_observed_at_both_endpoints',1)}\\% versus {rr('all_three_prices_observed_at_both_endpoints',0)}\\%)", 'measurement_robustness'),
 ('drop burial', '(75.4\\% versus 22.9\\%)', f"({rr('exclude_minimum_casket_burial_measure',1)}\\% versus {rr('exclude_minimum_casket_burial_measure',0)}\\%)", 'measurement_robustness'),
 ('5% threshold', '(69.9\\% versus 19.7\\%)', f"({p1(thr[thr.threshold.eq('More than 5 percent')&thr.date_change.eq(1)].rate.iloc[0])}\\% versus {p1(thr[thr.threshold.eq('More than 5 percent')&thr.date_change.eq(0)].rate.iloc[0])}\\%)", 'headline_thresholds'),
 ('worst-case gap', '26.4 percentage points', f"{100*bnd.lower_bound_gap.iloc[0]:.1f} percentage points", 'attrition_gap_bounds'),
 ('fee same-date', '2 of 112', f"{r['component_rates_all_annual_links']['ndf']['same_chg']} of {r['component_rates_all_annual_links']['ndf']['same_n']}", 'component rates'),
 ('J CI', '0.454--0.681', f"{sb['J_ci95'][0]:.3f}--{sb['J_ci95'][1]:.3f}", 'sim_boot J_ci95'),
 ('gamma_D FE', '0.088 (0.038)', f"{A('(1) date change on prior age'):.3f} ({A('(1) date change on prior age','se'):.3f})", 'list_age_tests (1)'),
 ('gamma_R FE', '0.045 (0.022)', f"{A('(2) price change on prior age'):.3f} ({A('(2) price change on prior age','se'):.3f})", 'list_age_tests (2)'),
 ('J*gamma_R', 'J\\times0.045=0.025', f"J\\times{A('(2) price change on prior age'):.3f}={r['J']*A('(2) price change on prior age'):.3f}", 'computed'),
 ('ratio', '3.5 times', f"{A('(1) date change on prior age')/(r['J']*A('(2) price change on prior age')):.1f} times", 'computed'),
 ('diff CI', '(0.006--0.138)', f"({sb['gamma_D_minus_J_gamma_R_ci95'][0]:.3f}--{sb['gamma_D_minus_J_gamma_R_ci95'][1]:.3f})", 'sim_boot'),
 ('gamma_D year FE', 'reverses sign ($-0.076$)', f"reverses sign (${A('(1y) date change on prior age, year FE'):.3f}$)", 'list_age_tests (1y)'),
 ('gamma_R year FE', 'near zero ($-0.008$)', f"near zero (${A('(2y) price change on prior age, year FE'):.3f}$)", 'list_age_tests (2y)'),
 ('D|R year FE', '($-0.072$, wild $p=0.016$)', f"(${A('(4y) date change on age given price, year FE'):.3f}$, wild $p={A('(4y) date change on age given price, year FE','wild_p'):.3f}$)", 'list_age_tests (4y)'),
 ('intervals per unit', 'about seven intervals', 'about seven intervals' if round(T['intervals_per_unit_mean']) == 7 else 'FAIL', 'panel_T'),
 ('sim within range', '0.063--0.091', f"{min(sb[k]['unit_year_FE_mean'] for k in sb if k.startswith('sim_') and isinstance(sb[k],dict)):.3f}--{max(sb[k]['unit_year_FE_mean'] for k in sb if k.startswith('sim_') and isinstance(sb[k],dict)):.3f}", 'sim_boot'),
 ('sim pooled range', '$-0.101$ to 0.003', f"${min(sb[k]['year_FE_mean'] for k in sb if k.startswith('sim_') and isinstance(sb[k],dict)):.3f}$ to {max(sb[k]['year_FE_mean'] for k in sb if k.startswith('sim_') and isinstance(sb[k],dict)):.3f}", 'sim_boot'),
 ('KC new-month rates', '(62.7\\% and 79.4\\%)', f"({p1(kc.loc['2018-2020','rate_new'])}\\% and {p1(kc.loc['2020-2023','rate_new'])}\\%)", 'kc_market_specific_associations'),
 ('KC same-month rates', '(4.5\\% and 14.3\\%, from only 22 and 7 intervals)', f"({p1(kc.loc['2018-2020','rate_same'])}\\% and {p1(kc.loc['2020-2023','rate_same'])}\\%, from only {kc.loc['2018-2020','same_n']} and {kc.loc['2020-2023','same_n']} intervals)", 'kc'),
 ('KC J', '$J$ is 0.462 and 0.282', f"$J$ is {f3(kc.loc['2018-2020','J'])} and {f3(kc.loc['2020-2023','J'])}", 'kc'),
 ('Austin DC 2yr J', 'gives $J=0.303$', f"gives $J={f3(ap['A1_by_measure']['Direct cremation, two-year links']['J'])}$", 'appendix A1'),
 ('decreases', '(18 of 49 versus 55 of 403)', f"({ap['A2_noise']['same']['decreases']} of {ap['A2_noise']['same']['events']} versus {ap['A2_noise']['new']['decreases']} of {ap['A2_noise']['new']['events']})", 'appendix A2'),
 ('reversals', '(6 of 40 versus 12 of 315)', f"({ap['A2_noise']['same']['reverts']} of {ap['A2_noise']['same']['next_year_observed']} versus {ap['A2_noise']['new']['reverts']} of {ap['A2_noise']['new']['next_year_observed']})", 'appendix A2'),
 ('burial decreases', '12 of its 22 same-date changes are decreases and 5 of 19 reverse', (lambda b: f"{b['decreases']} of its {b['events']} same-date changes are decreases and {b['reverts']} of {b['observed']} reverse")(ap['A2_noise']['same']['by_component']['immediate_burial']), 'appendix A2'),
 ('cremation increases', '(20 of 26 are increases and 1 of 20 reverses)', (lambda b: f"({b['events']-b['decreases']} of {b['events']} are increases and {b['reverts']} of {b['observed']} reverses)")(ap['A2_noise']['same']['by_component']['direct_cremation']), 'appendix A2'),
 ('persistent', '85.8\\% versus 28.7\\% ($J=0.578$)', (lambda q: f"{p1(q['rate_new'])}\\% versus {p1(q['rate_same'])}\\% ($J={f3(q['J'])}$)")(ap['A2_noise']['persistent_keep_unverifiable']), 'appendix A2'),
 ('abstract annual J 0.570 vs 0.303', 'against 0.570 for annual links', f"against {f3(r['J'])} for annual links", 'el_results'),
 ('fee unchanged', '110 of 112 links', f"{r['component_rates_all_annual_links']['ndf']['same_n']-r['component_rates_all_annual_links']['ndf']['same_chg']} of {r['component_rates_all_annual_links']['ndf']['same_n']} links", 'component rates'),
 ('KC carried entries', '25 of 113 entries', '25 of 113 entries', 'FCA-GKC 2020 publication (not recomputed here)'),
 ('FCAFL 2019 list', 'reports a list dated 2019', 'reports a list dated 2019', 'FCAFL 2025 survey, Table 1, Perkins 10 Feb. 2019 (not recomputed here)'),
 ('misclassified share', 'one interval in five', 'one interval in five' if abs((1-(r['new_date_reprice']+r['same_date_n']-r['same_date_reprice'])/r['n'])-0.2) < 0.01 else 'FAIL', '2x2 accuracy'),
 ('audit flags', 'All 45 intervals', 'All 45 intervals', 'audit ledger, austin_pipeline/source_results (not recomputed here)'),
 ('audit removed', 'removing five false flags', 'removing five false flags', 'audit ledger, austin_pipeline/source_results (not recomputed here)'),
]
rows = []; bad = 0
for lab, lit, comp, srcn in C:
    in_text = lit in tex_nb or lit in tex
    ok = in_text and lit == comp
    bad += not ok
    rows.append(dict(claim=lab, printed=lit, computed=comp, in_text=in_text, match=ok, source=srcn))
out = pd.DataFrame(rows); out.to_csv('output/number_checks.csv', index=False)
print(out.to_string(index=False)); print(f'\n{len(out)-bad}/{len(out)} checks pass')
sys.exit(1 if bad else 0)
