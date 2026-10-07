#!/usr/bin/env python3
"""(a) Block bootstrap (resampling establishment histories) for J, gamma_D, gamma_R and
gamma_D - J*gamma_R with 4,999 draws. (b) Constant-hazard simulation: how large are the
list-age coefficients when there is NO duration dependence? Writes output/sim_boot.json."""
import json, numpy as np, pandas as pd
from pathlib import Path
d = pd.read_csv(Path(__import__('os').environ.get('AUSTIN_PIPELINE','../austin_pipeline'))/'analytic_panel.csv').sort_values(['phone7', 'survey_year'])
# rebuild the 348-interval sample exactly as el_analysis.py does
src = open('el_analysis.py').read()
ns = {}
exec(src[:src.index("res = {}")], ns)
m = ns['m'][['phone7', 'survey_year', 'D', 'R', 'A']].reset_index(drop=True)
years = sorted(m.survey_year.unique())

def ols_coef(y, A, unit, yr):
    X = np.column_stack([A, pd.get_dummies(unit).values.astype(float), pd.get_dummies(yr).values.astype(float)[:, 1:]])
    return np.linalg.lstsq(X, y, rcond=None)[0][0]

groups = {u: g.index.values for u, g in m.groupby('phone7')}
ids = list(groups)
rng = np.random.default_rng(20260928)
out = []
for b in range(4999):
    pick = rng.choice(len(ids), len(ids))
    idx = np.concatenate([groups[ids[k]] for k in pick])
    unit = np.concatenate([np.full(len(groups[ids[k]]), j) for j, k in enumerate(pick)])
    s = m.loc[idx]
    D, R, A, yr = s.D.values, s.R.values, s.A.values, s.survey_year.values
    tp = ((D == 1) & (R == 1)).sum(); fn = ((D == 0) & (R == 1)).sum(); fp = ((D == 1) & (R == 0)).sum(); tn = ((D == 0) & (R == 0)).sum()
    J = tp / (tp + fn) + tn / (tn + fp) - 1
    gD = ols_coef(D, A, unit, yr); gR = ols_coef(R, A, unit, yr)
    out.append((J, gD, gR, gD - J * gR))
o = np.array(out)
res = {'bootstrap_draws': len(o)}
for k, nm in enumerate(['J', 'gamma_D', 'gamma_R', 'gamma_D_minus_J_gamma_R']):
    res[nm + '_ci95'] = [float(np.percentile(o[:, k], 2.5)), float(np.percentile(o[:, k], 97.5))]
res['share_draws_gamma_D_minus_J_gamma_R_le_0'] = float((o[:, 3] <= 0).mean())

# (b) constant-hazard simulation
def simulate(a, b, reps=500, N=53, T=7, seed=1):
    rng = np.random.default_rng(seed); fe = []; ye = []
    for _ in range(reps):
        U = []; Tt = []; Dv = []; Av = []
        for i in range(N):
            p = rng.beta(a, b) if b else a
            age = rng.uniform(0, 3)
            for t in range(1, T + 1):
                Dt = float(rng.random() < p)
                U.append(i); Tt.append(t); Dv.append(Dt); Av.append(age)
                age = rng.uniform(0, 1) if Dt == 1 else age + 1
        U, Tt, Dv, Av = map(np.array, (U, Tt, Dv, Av))
        fe.append(ols_coef(Dv, Av, U, Tt))
        X = np.column_stack([Av, pd.get_dummies(Tt).values.astype(float)])
        ye.append(np.linalg.lstsq(X, Dv, rcond=None)[0][0])
    return dict(unit_year_FE_mean=float(np.mean(fe)), unit_year_FE_sd=float(np.std(fe)),
                year_FE_mean=float(np.mean(ye)), year_FE_sd=float(np.std(ye)))
res['sim_heterogeneous_beta_2.6_1.4'] = simulate(2.6, 1.4)
res['sim_homogeneous_p_0.65'] = simulate(0.65, None)
res['sim_more_dispersed_beta_1.3_0.7'] = simulate(1.3, 0.7)
res['sim_note'] = ('Constant per-provider hazard (no duration dependence); 53 providers, 7 annual intervals; '
                   'initial list age U(0,3); a new list is U(0,1) years old at the next survey, otherwise age+1. '
                   'Hazard mean 0.65 matches Austin date-change frequency 0.649.')
json.dump(res, open('output/sim_boot.json', 'w'), indent=2)
print(json.dumps(res, indent=2))
