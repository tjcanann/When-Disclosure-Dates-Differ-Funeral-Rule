#!/usr/bin/env python3
"""Economics Letters analysis: verification of Austin headline numbers, the
list-age test with each outcome on its own, misclassification diagnostics, and
the Austin two-year-gap benchmark. Input: analytic_panel.csv written by ../austin_pipeline.
Pairs are rebuilt exactly as in worksheet_revision_analysis.py (348 intervals)."""
from pathlib import Path
import json, os, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

REP = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('AUSTIN_PIPELINE', '../austin_pipeline'))
KC_DIR = Path(os.environ.get('KC_DIR', '../kansas_city'))
OUT = Path('output'); OUT.mkdir(exist_ok=True)
d = pd.read_csv(REP/'analytic_panel.csv').sort_values(['phone7','survey_year']).reset_index(drop=True)
HEAD = ['direct_cremation','immediate_burial','ndf']

def build(data, gap):
    """Link each record to the same unit's record exactly `gap` calendar years earlier."""
    z = data.copy()
    z['gpl_date'] = pd.to_datetime(z.gpl_effective_date, errors='coerce')
    lag = z[['phone7','survey_year','gpl_date','gpl_age_days_clean']+HEAD].copy()
    lag['survey_year'] = lag.survey_year + gap
    lag = lag.rename(columns={c: 'L_'+c for c in lag.columns if c not in ('phone7','survey_year')})
    z = z.merge(lag, on=['phone7','survey_year'], how='left')
    has = z.gpl_date.notna() & z.L_gpl_date.notna()
    z['D'] = z.gpl_date.ne(z.L_gpl_date).astype(float).where(has)
    flags = []
    for c in HEAD:
        ok = z[c].notna() & z['L_'+c].notna()
        f = (z[c]-z['L_'+c]).abs().gt(5).astype(float).where(ok)
        z['chg_'+c] = f; flags.append(f)
    F = pd.concat(flags, axis=1)
    z['n_common'] = F.notna().sum(axis=1)
    z['R'] = F.max(axis=1).where(z.n_common.ge(2))
    z['R_noburial'] = pd.concat([z.chg_direct_cremation, z.chg_ndf], axis=1).max(axis=1).where(
        z[['chg_direct_cremation','chg_ndf']].notna().any(axis=1))
    return z

z1 = build(d, 1)
# prior list age = age of the list observed in the preceding annual survey (as in the paper)
z1['A'] = z1.L_gpl_age_days_clean/365.25
m = z1[z1.D.notna() & z1.R.notna() & z1.n_common.ge(2)].copy()
assert len(m) == 348, len(m)
assert tuple(m.groupby('D').R.sum()) == (40, 197) and tuple(m.groupby('D').size()) == (122, 226)
assert m.A.notna().all()

res = {}
tp, fn = int(((m.D==1)&(m.R==1)).sum()), int(((m.D==0)&(m.R==1)).sum())
fp, tn = int(((m.D==1)&(m.R==0)).sum()), int(((m.D==0)&(m.R==0)).sum())
sens, spec = tp/(tp+fn), tn/(tn+fp)
res.update(n=len(m), new_date_n=tp+fp, new_date_reprice=tp, same_date_n=tn+fn, same_date_reprice=fn,
           rate_new=tp/(tp+fp), rate_same=fn/(fn+tn), raw_gap=tp/(tp+fp)-fn/(fn+tn),
           sensitivity=sens, specificity=spec, J=sens+spec-1,
           freq_date=m.D.mean(), freq_price=m.R.mean(), units=m.phone7.nunique(),
           years=f"{int(m.survey_year.min()-1)}-{int(m.survey_year.max())}")

def fit(formula, data, var, cluster='phone7'):
    f = smf.ols(formula, data).fit(cov_type='cluster', cov_kwds={'groups': data[cluster]})
    return dict(coef=f.params[var], se=f.bse[var], p=f.pvalues[var], n=int(f.nobs), clusters=data[cluster].nunique())

def wild_p(formula_r, formula_u, data, var, B=9999, seed=20260928, cluster='phone7'):
    """Restricted wild-cluster (Rademacher) bootstrap p-value for H0: var=0."""
    rng = np.random.default_rng(seed)
    fu = smf.ols(formula_u, data).fit(cov_type='cluster', cov_kwds={'groups': data[cluster]})
    t0 = fu.params[var]/fu.bse[var]
    fr = smf.ols(formula_r, data).fit()
    yhat, e = fr.fittedvalues.values, fr.resid.values
    g = pd.factorize(data[cluster])[0]; G = g.max()+1
    import patsy
    y, X = patsy.dmatrices(formula_u, data, return_type='dataframe')
    Xv = X.values; j = list(X.columns).index(var)
    XtXi = np.linalg.pinv(Xv.T@Xv)
    cnt = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], G)[g]
        ys = yhat + e*w
        b = XtXi@(Xv.T@ys); u = ys - Xv@b
        S = np.zeros((Xv.shape[1],)*2)
        sc = np.zeros((G, Xv.shape[1])); np.add.at(sc, g, Xv*u[:,None])
        n, k = Xv.shape; c = G/(G-1)*(n-1)/(n-k)
        V = c*XtXi@(sc.T@sc)@XtXi
        if abs(b[j]/np.sqrt(V[j,j])) >= abs(t0): cnt += 1
    return (cnt+1)/(B+1)

FE = ' + C(phone7) + C(survey_year)'
# cremation/fee sample as in adversarial_analysis.py: both prices comparable, any date-observed annual link
NB = z1[z1.D.notna() & z1.chg_direct_cremation.notna() & z1.chg_ndf.notna()].copy()
NB['A'] = NB.L_gpl_age_days_clean/365.25
NB = NB[NB.A.notna()]
assert tuple(NB.groupby('D').size()) == (109, 195) and tuple(NB.groupby('D').R_noburial.sum()) == (25, 147)
YR = ' + C(survey_year)'
rows = []
specs = [
 ('(1) date change on prior age',         'D ~ A'+FE,      'D ~ 1'+FE,  'A', m),
 ('(2) price change on prior age',        'R ~ A'+FE,      'R ~ 1'+FE,  'A', m),
 ('(3) price change on age, given date',  'R ~ D + A'+FE,  'R ~ D'+FE,  'A', m),
 ('(4) date change on age, given price',  'D ~ R + A'+FE,  'D ~ R'+FE,  'A', m),
 ('(5) price change on age, same-date',   'R ~ A'+FE,      'R ~ 1'+FE,  'A', m[m.D==0]),
 ('(6) price change on age, new-date',    'R ~ A'+FE,      'R ~ 1'+FE,  'A', m[m.D==1]),
 ('(1y) date change on prior age, year FE','D ~ A'+YR,     'D ~ 1'+YR,  'A', m),
 ('(2y) price change on prior age, year FE','R ~ A'+YR,    'R ~ 1'+YR,  'A', m),
 ('(4y) date change on age given price, year FE','D ~ R + A'+YR,'D ~ R'+YR,'A', m),
 ('(3y) price change on age given date, year FE','R ~ D + A'+YR,'R ~ D'+YR,'A', m),
 ('(2nb) cremation/fee change on age (both comparable)','R_noburial ~ A'+FE,'R_noburial ~ 1'+FE,'A', NB),
 ('(1nb) date change on age, cremation/fee sample','D ~ A'+FE,'D ~ 1'+FE,'A', NB),
]
for lab, fu, fr, var, data in specs:
    data = data.copy()
    r = fit(fu, data, var); r['wild_p'] = wild_p(fr, fu, data, var, B=4999)
    r['model'] = lab; rows.append(r)
# Headline date coefficients (verification)
for lab, f in [('year FE', 'R ~ D'+YR), ('unit+year FE', 'R ~ D'+FE), ('unit+year FE + prior age', 'R ~ D + A'+FE)]:
    r = fit(f, m, 'D'); r['model'] = 'headline: '+lab; rows.append(r)
age = pd.DataFrame(rows)[['model','coef','se','p','wild_p','n','clusters']]
age.to_csv(OUT/'list_age_tests.csv', index=False)

# Misclassification decomposition. Under non-differential misclassification,
# E[D|R,A]=a0+J*R does not depend on A, so gamma_D = J*gamma_R.
gR = age.set_index('model').loc['(2) price change on prior age','coef']
gD = age.set_index('model').loc['(1) date change on prior age','coef']
res.update(gamma_D=gD, gamma_R=gR, J_times_gamma_R=res['J']*gR, gamma_D_minus_J_gamma_R=gD-res['J']*gR,
           ratio_gamma_D_to_gamma_R=gD/gR if gR != 0 else np.nan)
# Bootstrap intervals for J and the age gradients: see sim_and_bootstrap.py (4,999 draws).

# Lagged-state symmetry test: does last year's date/price state predict this year's date/price change?
z1 = z1.sort_values(['phone7','survey_year'])
lagm = m[['phone7','survey_year','D','R']].copy(); lagm['survey_year'] += 1
lagm = lagm.rename(columns={'D':'D_lag','R':'R_lag'})
ml = m.merge(lagm, on=['phone7','survey_year'], how='inner')
sym = []
for y in ['D','R']:
    for x in ['D_lag','R_lag']:
        r = fit(f'{y} ~ {x} + C(survey_year)', ml, x); r['model'] = f'{y} on {x}, year FE'; sym.append(r)
pd.DataFrame(sym).to_csv(OUT/'lagged_state_tests.csv', index=False)
res['lagged_state_n'] = len(ml)

# Age-bin descriptive table: date-change and price-change rates by prior list age
m['A_bin'] = pd.cut(m.A, [-0.01, 0.5, 1.0, 1.5, 100], labels=['<=0.5y','0.5-1y','1-1.5y','>1.5y'])
bins = m.groupby('A_bin', observed=True).agg(n=('D','size'), date_rate=('D','mean'), price_rate=('R','mean'),
        same_date_n=('D', lambda s: int((s==0).sum())))
bins.to_csv(OUT/'age_bins.csv')

# Two-year-gap benchmark (comparable to the two- and three-year Kansas City windows)
z2 = build(d, 2)
m2 = z2[z2.D.notna() & z2.R.notna() & z2.n_common.ge(2)].copy()
two = {}
for lab, s in [('all_two_year_links', m2), ('window_years_2020_2022_2024', m2[m2.survey_year.isin([2022, 2024])])]:
    tp2, fn2 = int(((s.D==1)&(s.R==1)).sum()), int(((s.D==0)&(s.R==1)).sum())
    fp2, tn2 = int(((s.D==1)&(s.R==0)).sum()), int(((s.D==0)&(s.R==0)).sum())
    two[lab] = dict(n=len(s), new_n=tp2+fp2, new_reprice=tp2, same_n=tn2+fn2, same_reprice=fn2,
                    rate_new=tp2/(tp2+fp2) if tp2+fp2 else None, rate_same=fn2/(fn2+tn2) if fn2+tn2 else None,
                    sensitivity=tp2/(tp2+fn2) if tp2+fn2 else None, specificity=tn2/(tn2+fp2) if tn2+fp2 else None)
    if two[lab]['sensitivity'] is not None and two[lab]['specificity'] is not None:
        two[lab]['J'] = two[lab]['sensitivity']+two[lab]['specificity']-1
    if len(s) > 20:
        r = fit('R ~ D + C(survey_year)', s, 'D'); two[lab]['coef_year_FE'] = r['coef']; two[lab]['se'] = r['se']
res['two_year'] = two
# Component-level same-date change rates (which prices move without a date change?)
comp = {}
for c in HEAD:
    s = z1[z1.D.notna() & z1['chg_'+c].notna()]
    comp[c] = dict(same_n=int((s.D==0).sum()), same_chg=int(s.loc[s.D==0,'chg_'+c].sum()),
                   new_n=int((s.D==1).sum()), new_chg=int(s.loc[s.D==1,'chg_'+c].sum()))
res['component_rates_all_annual_links'] = comp

# Kansas City interim second market (published tables, month dates), from package outputs
kc = pd.read_csv(KC_DIR/'kc_market_specific_associations.csv')
kc = kc[kc.scenario.eq('strict_no_known_carry') & kc.threshold.eq('gt5')]
kcr = []
for _, r in kc.iterrows():
    tp_, fp_ = r.changed_month_reprices, r.changed_month_n - r.changed_month_reprices
    fn_, tn_ = r.unchanged_month_reprices, r.unchanged_month_n - r.unchanged_month_reprices
    se_, sp_ = tp_/(tp_+fn_), tn_/(tn_+fp_)
    kcr.append(dict(window=r.window, outcome=r.outcome, n=r.n, new_n=r.changed_month_n, new_reprice=tp_,
                    same_n=r.unchanged_month_n, same_reprice=fn_, rate_new=r.changed_month_rate,
                    rate_same=r.unchanged_month_rate, sensitivity=se_, specificity=sp_, J=se_+sp_-1))
pd.DataFrame(kcr).to_csv(OUT/'kc_signal_metrics.csv', index=False)

json.dump(res, open(OUT/'el_results.json','w'), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
print(age.to_string(index=False))
print(pd.DataFrame(sym).to_string(index=False))
print(bins.to_string())
print(pd.DataFrame(kcr).to_string(index=False))

# Direct test that the age gradient differs by proxy: regress (D - R) on age.
m['DmR'] = m.D - m.R
dif = []
for lab, f in [('unit+year FE', 'DmR ~ A'+FE), ('year FE', 'DmR ~ A'+YR)]:
    r = fit(f, m, 'A'); r['model'] = 'D minus R on age, '+lab
    r['wild_p'] = wild_p(f.replace('A +', '1 +').replace('~ A', '~ 1'), f, m.copy(), 'A', B=4999); dif.append(r)
pd.DataFrame(dif).to_csv(OUT/'proxy_difference_tests.csv', index=False)
print(pd.DataFrame(dif).to_string(index=False))
# units and intervals per unit (Nickell exposure)
print('intervals per unit: mean', m.groupby('phone7').size().mean(), 'median', m.groupby('phone7').size().median())
json.dump({'intervals_per_unit_mean': float(m.groupby('phone7').size().mean())}, open(OUT/'panel_T.json','w'))
