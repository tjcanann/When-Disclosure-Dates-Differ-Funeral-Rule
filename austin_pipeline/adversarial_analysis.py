#!/usr/bin/env python3
"""Additional measurement/inference checks for the adversarial CEP revision.
No causal estimates, new markets, or retrospective choice of headline outcomes.
Run after apply_source_corrections.py and the baseline scripts. Fixed seeds below.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import analysis

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'adversarial_output';OUT.mkdir(exist_ok=True)
d=pd.read_csv(ROOT/'analytic_panel.csv')
m=d[d.gap_years_clean.eq(1)&d.n_headline_comparable_clean.ge(2)&d.formal_revision_clean.notna()&d.any_headline_change_clean.notna()].copy()
m['prior_gpl_age_years']=m.prev_gpl_age_days_clean/365.25
# Stable coded brand blocks; other telephone identifiers remain singletons.
brand_map=d.dropna(subset=['brand_group']).groupby('phone7').brand_group.first().to_dict()
m['brand_block']=[brand_map.get(ph,f'unit_{ph}') for ph in m.phone7]
R='any_headline_change_clean';D='formal_revision_clean'
formulas={
 'year_FE':f'{R} ~ {D} + C(survey_year)',
 'establishment_year_FE':f'{R} ~ {D} + C(phone7) + C(survey_year)',
 'establishment_year_FE_prior_age':f'{R} ~ {D} + prior_gpl_age_years + C(phone7) + C(survey_year)'}

def wild(formula,data,cluster_col,B=4999,seed=20260907):
 y,X=patsy.dmatrices(formula,data,return_type='dataframe');yv=np.asarray(y).ravel();xx=np.asarray(X);idx=list(X.columns).index(D)
 codes,uni=pd.factorize(data.loc[X.index,cluster_col]);G=len(uni); n,k=xx.shape
 assert np.linalg.matrix_rank(xx)==k
 pinv=np.linalg.pinv(xx); b=pinv@yv; res=yv-xx@b
 h=pinv[idx,:];agg=np.zeros((G,n));agg[codes,np.arange(n)]=1
 fac=(G/(G-1))*((n-1)/(n-k));se=np.sqrt(fac*np.sum((agg@(h*res))**2));t=b[idx]/se
 xr=np.delete(xx,idx,axis=1);fit=xr@np.linalg.lstsq(xr,yv,rcond=None)[0];ur=yv-fit
 rng=np.random.default_rng(seed);ex=0
 for start in range(0,B,200):
  size=min(200,B-start);w=rng.choice([-1.,1.],size=(G,size));ys=fit[:,None]+ur[:,None]*w[codes,:]
  bs=pinv@ys;rs=ys-xx@bs;ss=np.sqrt(fac*np.sum((agg@(h[:,None]*rs))**2,axis=0))
  ex+=int(np.sum(np.abs(bs[idx,:]/np.maximum(ss,1e-12))>=abs(t)))
 return {'coefficient':b[idx],'se':se,'wild_p':(ex+1)/(B+1),'exceedances':ex,'B':B,'n':n,'clusters':G}
rows=[]
for j,(name,formula) in enumerate(formulas.items()):
 for col in ['phone7','brand_block']:
  x=wild(formula,m,col,seed=20260907+j)
  check=smf.ols(formula,m).fit(cov_type='cluster',cov_kwds={'groups':m[col]})
  assert np.isclose(x['se'],check.bse[D],atol=1e-10)
  rows.append({'specification':name,'cluster_level':col,**x})
pd.DataFrame(rows).to_csv(OUT/'models_and_cluster_inference.csv',index=False)

# Resample whole observed histories, not dependent intervals as independent trials.
metrics={
 'sensitivity':(lambda a:a[:,3],lambda a:a[:,3]+a[:,1]),
 'specificity':(lambda a:a[:,0],lambda a:a[:,0]+a[:,2]),
 'positive_predictive_value':(lambda a:a[:,3],lambda a:a[:,3]+a[:,2]),
 'negative_predictive_value':(lambda a:a[:,0],lambda a:a[:,0]+a[:,1]),
 'accuracy':(lambda a:a[:,0]+a[:,3],lambda a:a.sum(axis=1)),
 'same_date_repricing_rate':(lambda a:a[:,1],lambda a:a[:,0]+a[:,1])}
cell=(2*m[D].astype(int)+m[R].astype(int));counts=np.bincount(cell,minlength=4).reshape(1,4)
cirows=[]
for col in ['phone7','brand_block']:
 cells=pd.crosstab(m[col],cell).reindex(columns=range(4),fill_value=0).to_numpy();G=len(cells);rng=np.random.default_rng(20260907)
 weights=rng.multinomial(G,np.repeat(1/G,G),size=9999);sim=weights@cells
 for name,(num,den) in metrics.items():
  ratio=num(sim)/den(sim);n=int(num(counts)[0]);N=int(den(counts)[0]);lo,hi=proportion_confint(n,N,method='wilson')
  cirows.append({'metric':name,'cluster_level':col,'numerator':n,'denominator':N,'estimate':n/N,'cluster_percentile_lo':np.quantile(ratio,.025),'cluster_percentile_hi':np.quantile(ratio,.975),'wilson_lo':lo,'wilson_hi':hi,'B':9999,'clusters':G})
pd.DataFrame(cirows).to_csv(OUT/'signal_intervals_clustered.csv',index=False)

# Balanced observability, narrower price basket, source exclusions, equal-unit weights.
results=[]
def rates(sample,z,outcome=R,weight=None):
 for val in [0,1]:
  q=z[z[D].eq(val)&z[outcome].notna()];w=None if weight is None else q[weight]
  results.append({'sample':sample,'gpl_date_change':val,'n':len(q),'repricing_count':int(q[outcome].sum()),'rate':float(q[outcome].mean() if w is None else np.average(q[outcome],weights=w)),'outcome':outcome})
rates('baseline_three_headline_prices_at_least_two_observed',m)
rates('all_three_prices_observed_at_both_endpoints',m[m.n_headline_comparable_clean.eq(3)])
noib=d[d.gap_years_clean.eq(1)&d[D].notna()&d.price_change_direct_cremation_clean.notna()&d.price_change_ndf_clean.notna()].copy()
noib['cremation_or_basic_fee_change']=noib[['price_change_direct_cremation_clean','price_change_ndf_clean']].max(axis=1)
rates('exclude_minimum_casket_burial_measure',noib,'cremation_or_basic_fee_change')
for c in ['direct_cremation','immediate_burial','ndf']:
 q=d[d.gap_years_clean.eq(1)&d[D].notna()]
 rates('component_'+c,q,'price_change_'+c+'_clean')
web=m[m.phone7.eq(2372365)&m.survey_year.isin([2023,2024])].index
rates('exclude_two_previously_flagged_website_repricing_intervals',m.drop(web))
# Explicit identified source-mode caveat; not a claim all channels are known.
extra=m[m.phone7.eq(3982343)&m.survey_year.eq(2017)].index
rates('exclude_flagged_website_intervals_and_2017_phone_clarification',m.drop(web.union(extra)))
rates('exclude_entire_Gipson_and_Eeds_histories',m[~m.phone7.isin([2372365,3982343])])
rates('both_endpoint_surveys_2019_or_later',m[m.prev_survey_year_clean.ge(2019)])
rates('omit_2016_to_2017_transition',m[~m.survey_year.eq(2017)])
u=m.copy();u['unit_weight']=1/u.groupby('phone7')[R].transform('size');rates('equal_weight_each_telephone_identifier',u,weight='unit_weight')
pd.DataFrame(results).to_csv(OUT/'measurement_robustness.csv',index=False)

# Matched lead/lag support: require BOTH immediately neighboring intervals.
it=d[d.gap_years_clean.eq(1)&d[D].notna()];lookup={(int(r.phone7),int(r.survey_year)):getattr(r,D) for r in it.itertuples()}
sd=m[m[D].eq(0)].copy()
for key,offset in [('previous',-1),('next',1)]:sd[key]=[lookup.get((int(ph),int(y)+offset),np.nan) for ph,y in zip(sd.phone7,sd.survey_year)]
sd['both_observed']=sd.previous.notna()&sd.next.notna();q=sd[sd.both_observed].copy();q['any_adjacent']=(q.previous.eq(1)|q.next.eq(1)).astype(float)
trows=[]
for val in [0,1]:
 s=q[q[R].eq(val)];trows.append({'same_date_repricing':val,'n':len(s),'previous_rate':s.previous.mean(),'next_rate':s.next.mean(),'any_adjacent_count':int(s.any_adjacent.sum()),'any_adjacent_rate':s.any_adjacent.mean()})
pd.DataFrame(trows).to_csv(OUT/'complete_window_timing.csv',index=False)
fit=smf.ols(f'any_adjacent ~ {R} + C(survey_year)',q).fit(cov_type='cluster',cov_kwds={'groups':q.phone7})
ci=fit.conf_int().loc[R]
pd.DataFrame([{'n':len(q),'clusters':q.phone7.nunique(),'coefficient':fit.params[R],'se':fit.bse[R],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1],'p':fit.pvalues[R]}]).to_csv(OUT/'complete_window_timing_regression.csv',index=False)
q.to_csv(OUT/'complete_window_timing_panel.csv',index=False)

# Audit reconciliation and selected re-verification: assertions against saved source readings.
a=pd.read_csv(ROOT/'source_results'/'sticky_phase2b_initial_silent_pair_audit.csv');s=m[m[D].eq(0)&m[R].eq(1)]
a=a.merge(s[['phone7','survey_year']],on=['phone7','survey_year'],how='outer',indicator=True)
a['final_same_date_case']=a['_merge'].isin(['both','right_only']);a['entered_after_date_correction']=a['_merge'].eq('right_only');a.to_csv(OUT/'complete_same_date_audit_reconciliation.csv',index=False)
assert len(a[a.final_same_date_case])==40 and a.entered_after_date_correction.sum()==1
ver=pd.read_csv(ROOT/'inputs'/'visual_reverification.csv')
for r in ver.itertuples():
 x=d[d.phone7.eq(r.phone7)&d.survey_year.eq(r.survey_year)].iloc[0]
 for c in ['direct_cremation','immediate_burial','ndf']:
  exp=getattr(r,'verified_'+c);exp_prev=getattr(r,'verified_prev_'+c)
  if pd.notna(exp):assert np.isclose(x[c],exp)
  if pd.notna(exp_prev):assert np.isclose(x[f'prev_{c}_clean'],exp_prev)
 assert x[D]==0 and x[R]==1
ver.to_csv(OUT/'visual_reverification.csv',index=False)

# Recompute supporting persistence after corrected price-cell input.
h=analysis.build_component_hazards(d);state,formal,actual=analysis.state_dependence(d,h)
state.to_csv(OUT/'supporting_persistence.csv',index=False)
analysis.threshold_sensitivity(d).to_csv(OUT/'price_spell_thresholds.csv',index=False)
_,iy=analysis.inflation_year_diagnostics(d,h);iy.to_csv(OUT/'inflation_year_diagnostic.csv',index=False)
# Explicit denominators for administrative dates (often differ from all price-pair N).
sync=analysis.build_sync(d);sr=[]
for label in ['same_current_gpl_exact','same_prev_gpl_exact']:
 for same in [False,True]:
  v=sync.loc[sync.same_brand.eq(same),label].dropna();sr.append({'metric':label,'same_brand':same,'n':len(v),'matches':int(v.sum()),'rate':v.mean()})
pd.DataFrame(sr).to_csv(OUT/'date_synchronization_denominators.csv',index=False)
# Baseline label reshuffle measures only within-label agreement, NOT the gap.
perm,_=analysis.permutation_benchmark_group(d,'brand_group',nperm=1999)
perm.to_csv(OUT/'descriptive_label_benchmark.csv',index=False)

summary={'main_n':len(m),'price_cells_corrected':3,'main_binary_outcomes_changed':0,'baseline_price_panel_sha256':hashlib.sha256((ROOT/'inputs'/'analytic_panel_pre_adversarial.csv').read_bytes()).hexdigest(),'corrected_price_panel_sha256':hashlib.sha256((ROOT/'analytic_panel.csv').read_bytes()).hexdigest(),'random_rechecked_intervals':10,'purposive_rechecked_intervals':2,'rechecked_intervals_with_retained_flag':12,'rechecked_intervals_requiring_level_correction':1,'cluster_blocks':m.brand_block.nunique(),'scripts_compute_from_audited_panel_not_automated_pdf_reextraction':True}
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2))
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(trows).to_string(index=False));print(pd.DataFrame(results).to_string(index=False));print(summary)

# Recompute every reported headline-threshold and period row from current corrected data.
tr=[]
for label,mode,cut in [('More than $5','abs',5),('More than $25','abs',25),('More than 2.5 percent','pct',.025),('More than 5 percent','pct',.05)]:
 change=[]
 for c in ['direct_cremation','immediate_burial','ndf']:
  delta=m[f'delta_{c}_clean'];prev=m[f'prev_{c}_clean']
  a=delta.abs() if mode=='abs' else delta.abs()/prev
  change.append((a>cut).where(a.notna()))
 flag=pd.concat(change,axis=1).max(axis=1)
 for v in [0,1]:
  ix=m[D].eq(v);tr.append({'threshold':label,'date_change':v,'n':int(ix.sum()),'repricing_count':int(flag[ix].sum()),'rate':flag[ix].mean()})
pd.DataFrame(tr).to_csv(OUT/'headline_thresholds.csv',index=False)
pr=[]
for label,mask in [('Through 2019',m.survey_year.le(2019)),('Through 2020',m.survey_year.le(2020)),('2021-2024',m.survey_year.ge(2021)),('2022-2024',m.survey_year.ge(2022))]:
 for v in [0,1]:
  q=m[mask&m[D].eq(v)];pr.append({'period':label,'date_change':v,'n':len(q),'repricing_count':int(q[R].sum()),'rate':q[R].mean()})
pd.DataFrame(pr).to_csv(OUT/'period_rates.csv',index=False)
assert d.dropna(subset=['brand_group']).groupby('phone7').brand_group.nunique().max()==1
