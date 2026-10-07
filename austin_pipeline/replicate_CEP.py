#!/usr/bin/env python3
"""Replication checks for Price-List Dates and Repricing under the Funeral Rule.

Input: analytic_panel.csv (source-audited establishment-year panel)
Outputs: cep_main_results.csv and cep_sync_results.csv

The script reproduces the headline descriptive rates, fixed-effects association,
component change rates, and chain synchronization summaries used in the CEP manuscript.
The more extensive Phase 2C script is included as analysis.py for appendix diagnostics,
permutation benchmarks, persistence, inflation, and decomposition analyses.
"""
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
import analysis

HERE = Path(__file__).resolve().parent
PANEL = HERE / 'analytic_panel.csv'
OUT = HERE / 'replication_output'
OUT.mkdir(exist_ok=True)

df = pd.read_csv(PANEL)

# Clean consecutive-year headline sample.
main = df[(df['gap_years_clean'] == 1) &
          (df['n_headline_comparable_clean'] >= 2) &
          df['formal_revision_clean'].notna() &
          df['any_headline_change_clean'].notna()].copy()

rows = []
for dval, label in [(0.0, 'same_gpl_date'), (1.0, 'new_gpl_date')]:
    z = main[main['formal_revision_clean'].eq(dval)]
    rows.append({'result':'headline_repricing_rate','group':label,'n':len(z),
                 'estimate':z['any_headline_change_clean'].mean()})

for c, label in [('price_change_direct_cremation_clean','direct_cremation'),
                 ('price_change_immediate_burial_clean','immediate_burial'),
                 ('price_change_ndf_clean','basic_services_fee')]:
    base = df[(df['gap_years_clean'].eq(1)) & df['formal_revision_clean'].notna() & df[c].notna()].copy()
    for dval, g in [(None,'all'),(0.0,'same_gpl_date'),(1.0,'new_gpl_date')]:
        z = base if dval is None else base[base['formal_revision_clean'].eq(dval)]
        rows.append({'result':f'component_change_rate_{label}','group':g,'n':len(z),'estimate':z[c].mean()})

m_year = smf.ols('any_headline_change_clean ~ formal_revision_clean + C(survey_year)', data=main).fit(
    cov_type='cluster', cov_kwds={'groups':main['phone7']})
m_fe = smf.ols('any_headline_change_clean ~ formal_revision_clean + C(phone7) + C(survey_year)', data=main).fit(
    cov_type='cluster', cov_kwds={'groups':main['phone7']})
for name, m in [('year_FE','year_FE'),('establishment_and_year_FE','est_year_FE')]:
    model = m_year if m == 'year_FE' else m_fe
    rows.append({'result':'formal_revision_coefficient','group':name,'n':int(model.nobs),
                 'estimate':model.params['formal_revision_clean'],
                 'se':model.bse['formal_revision_clean'],
                 'p':model.pvalues['formal_revision_clean']})

pd.DataFrame(rows).to_csv(OUT/'cep_main_results.csv', index=False)

sync = analysis.build_sync(df)
srows=[]
samples={
    'all_pairs': pd.Series(True,index=sync.index),
    'active_pairs': sync['any_pair_active'].eq(1),
    'active_different_starting_prices': sync['any_pair_active'].eq(1) & sync['prior_price_vector_different'].eq(1),
}
for sample,mask in samples.items():
    for same,label in [(True,'same_brand'),(False,'different_brand')]:
        z=sync[mask & sync['same_brand'].eq(same)]
        if sample=='active_different_starting_prices' and not same:
            z=z[~z['same_owner']]
            label='different_owner_different_brand'
        srows.append({'sample':sample,'group':label,'n':len(z),
                      'component_vector_match':z['exact_component_change_vector'].mean(),
                      'dollar_vector_match':z['exact_dollar_delta_vector'].mean(),
                      'same_current_gpl_exact':z['same_current_gpl_exact'].mean(),
                      'same_prev_gpl_exact':z['same_prev_gpl_exact'].mean()})

pd.DataFrame(srows).to_csv(OUT/'cep_sync_results.csv', index=False)
print(f'Main sample N = {len(main)}')
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(srows).to_string(index=False))
