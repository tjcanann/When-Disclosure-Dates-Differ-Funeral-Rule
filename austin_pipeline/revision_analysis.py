from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import analysis
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'revision_output'
OUT.mkdir(exist_ok=True)
PANEL = pd.read_csv(ROOT / 'analytic_panel.csv')
SYNC = analysis.build_sync(PANEL)
AUDIT = pd.read_csv(ROOT / 'source_results' / 'sticky_phase2b_initial_silent_pair_audit.csv')

# County coding based on physical addresses and explicit facility-location footnotes.
# Heart of Texas 2437277: 2015 p8 note14 and 2016 p7 note14 identify Hays County despite Austin mailing address.
COUNTY = {
2228000:'Travis',2372365:'Bastrop',2372425:'Bastrop',2373836:'Bastrop',2437277:'Hays',
2443772:'Williamson',2513500:'Travis',2514118:'Travis',2561220:'Hays',2591610:'Williamson',
2608800:'Williamson',2688200:'Hays',2812340:'Bastrop',2815697:'Bastrop',2941499:'Travis',
3089188:'Bastrop',3212576:'Bastrop',3230122:'Travis',3268878:'Travis',3351155:'Travis',
3398878:'Travis',3523636:'Williamson',3525909:'Williamson',3534311:'Hays',3542509:'Travis',
3656000:'Williamson',3766200:'Caldwell',3826543:'Hays',3923676:'Hays',3962300:'Hays',
3982300:'Caldwell',3982343:'Caldwell',3984791:'Caldwell',4197224:'Travis',4419738:'Travis',
4421446:'Travis',4431366:'Travis',4443355:'Travis',4528811:'Travis',4545611:'Travis',
4721934:'Travis',4764355:'Travis',4769128:'Travis',4769164:'Travis',4777495:'Travis',
4784300:'Travis',8472226:'Hays',8511236:'Travis',8632564:'Williamson',8697775:'Williamson',
8698888:'Williamson',8755713:'Caldwell',8921172:'Travis'
}
pd.DataFrame([{'phone7':k,'county':v} for k,v in sorted(COUNTY.items())]).to_csv(OUT/'county_crosswalk.csv',index=False)

d = PANEL.copy()
d['chain_affiliated'] = d['brand_group'].notna().astype(int)
final = d[d['gap_years_clean'].eq(1) & d['formal_revision_clean'].notna() & d['any_headline_change_clean'].notna() & (d['n_headline_comparable_clean'] >= 2)].copy()
final['county'] = final['phone7'].map(COUNTY)

# Sample composition
unique_final = final[['phone7','brand_group','county']].drop_duplicates('phone7')
brand_sizes = unique_final.dropna(subset=['brand_group']).groupby('brand_group')['phone7'].nunique()
annual_counts = d.groupby('survey_year')['phone7'].nunique()
comp = [
    ('Establishment-year observations', len(d), ''),
    ('Unique establishment identifiers', d.phone7.nunique(), ''),
    ('Annual establishments observed, minimum', int(annual_counts.min()), ''),
    ('Annual establishments observed, maximum', int(annual_counts.max()), ''),
    ('Establishment-years with usable GPL date', int(d.gpl_effective_date.notna().sum()), f"{d.gpl_effective_date.notna().mean():.1%}"),
    ('Date-defined consecutive intervals', int(d.formal_revision_clean.notna().sum()), ''),
    ('Final headline intervals', len(final), f"{len(final)/d.formal_revision_clean.notna().sum():.1%} of date-defined intervals"),
    ('Establishments in final sample', final.phone7.nunique(), ''),
    ('Establishments assigned to coded multi-establishment brands', int(unique_final.brand_group.notna().sum()), f"{unique_final.brand_group.notna().mean():.1%}"),
    ('Multi-establishment brands in final sample', int(brand_sizes.size), ''),
    ('Mean establishments per multi-establishment brand', float(brand_sizes.mean()), ''),
]
pd.DataFrame(comp,columns=['statistic','value','note']).to_csv(OUT/'sample_composition.csv',index=False)

county = (final.groupby('county').agg(intervals=('phone7','size'), establishments=('phone7','nunique')).reset_index())
county['interval_share'] = county['intervals']/len(final)
county.to_csv(OUT/'county_composition.csv',index=False)
brand_sizes.rename('establishments').reset_index().to_csv(OUT/'brand_composition.csv',index=False)

# Establishment-year date-observability comparison.
d['has_usable_date'] = d['gpl_effective_date'].notna().astype(int)
d['n_headline_prices_current'] = d[['direct_cremation','immediate_burial','ndf']].notna().sum(axis=1)
date_rows=[]
for val,label in [(1,'Usable GPL date'),(0,'Missing GPL date')]:
    s=d[d.has_usable_date==val]
    date_rows.append({
        'group':label, 'n_establishment_years':len(s),
        'chain_share':s.chain_affiliated.mean(), 'mean_survey_year':s.survey_year.mean(),
        'mean_headline_prices_observed':s.n_headline_prices_current.mean(),
        'share_2plus_headline_prices':(s.n_headline_prices_current>=2).mean()
    })
pd.DataFrame(date_rows).to_csv(OUT/'date_observability_comparison.csv',index=False)

# Attrition / missingness among date-defined consecutive intervals.
z = d[d['formal_revision_clean'].notna()].copy()
z['included'] = (z['n_headline_comparable_clean'] >= 2).astype(int)
lag = d.sort_values(['phone7','survey_year'])[['phone7','survey_year','any_headline_change_clean']].copy()
lag['prior_observed_repricing'] = lag.groupby('phone7')['any_headline_change_clean'].shift(1)
z = z.merge(lag[['phone7','survey_year','prior_observed_repricing']],on=['phone7','survey_year'],how='left')
attr=[]
for label,mask in [('Included in headline sample',z.included==1),('Excluded: fewer than two comparable headline prices',z.included==0)]:
    s=z[mask]
    attr.append({
        'group':label,'n_intervals':len(s),'chain_share':s.chain_affiliated.mean(),
        'new_gpl_date_share':s.formal_revision_clean.mean(),'mean_survey_year':s.survey_year.mean(),
        'prior_repricing_share':s.prior_observed_repricing.mean(),'prior_repricing_n':s.prior_observed_repricing.notna().sum()
    })
pd.DataFrame(attr).to_csv(OUT/'attrition_comparison.csv',index=False)

# Worst-case Manski-style bounds over the 435 date-defined intervals.
bounds=[]
for rev,label in [(1,'New GPL date'),(0,'Same GPL date')]:
    s=z[z.formal_revision_clean==rev]
    obs=s[s.included==1]
    miss=s[s.included==0]
    y=obs.any_headline_change_clean.sum()
    bounds.append({'gpl_status':label,'observed_n':len(obs),'missing_price_outcome_n':len(miss),
                   'observed_rate':obs.any_headline_change_clean.mean(),
                   'lower_bound_rate':y/len(s),'upper_bound_rate':(y+len(miss))/len(s)})
bdf=pd.DataFrame(bounds)
bdf.to_csv(OUT/'attrition_worst_case_bounds.csv',index=False)
new=bdf[bdf.gpl_status=='New GPL date'].iloc[0]; same=bdf[bdf.gpl_status=='Same GPL date'].iloc[0]
pd.DataFrame([{'lower_bound_gap':new.lower_bound_rate-same.upper_bound_rate,
               'upper_bound_gap':new.upper_bound_rate-same.lower_bound_rate}]).to_csv(OUT/'attrition_gap_bounds.csv',index=False)

# IPW reweighting to the date-defined interval universe on observables.
z['chain_affiliated'] = z.brand_group.notna().astype(int)
logit = smf.logit('included ~ chain_affiliated + formal_revision_clean + C(survey_year)', data=z).fit(disp=0)
inc=z[z.included==1].copy(); inc['p_include']=logit.predict(inc); inc['ipw']=1/inc.p_include
ipw=[]
for rev,label in [(1,'New GPL date'),(0,'Same GPL date')]:
    s=inc[inc.formal_revision_clean==rev]
    ipw.append({'gpl_status':label,'n':len(s),'unweighted_rate':s.any_headline_change_clean.mean(),
                'ipw_rate':np.average(s.any_headline_change_clean,weights=s.ipw)})
pd.DataFrame(ipw).to_csv(OUT/'attrition_ipw.csv',index=False)

# High-coverage establishment restrictions.
coverage=d.assign(date_ok=d.gpl_effective_date.notna()).groupby('phone7').date_ok.mean()
pairs_per_unit=final.groupby('phone7').size()
restr=[]
for name, phones in [
    ('GPL date observed in >=80% of establishment-years', coverage[coverage>=.80].index),
    ('At least 5 usable headline intervals', pairs_per_unit[pairs_per_unit>=5].index),
]:
    s=final[final.phone7.isin(phones)]
    for rev,label in [(1,'New GPL date'),(0,'Same GPL date')]:
        q=s[s.formal_revision_clean==rev]
        restr.append({'restriction':name,'gpl_status':label,'n':len(q),'repricing_rate':q.any_headline_change_clean.mean(),'n_establishments':s.phone7.nunique()})
pd.DataFrame(restr).to_csv(OUT/'coverage_restrictions.csv',index=False)

# Signal metrics + Wilson 95 CIs from the 2x2.
cm = pd.crosstab(final.formal_revision_clean.astype(int), final.any_headline_change_clean.astype(int))
tp=int(cm.loc[1,1]); fp=int(cm.loc[1,0]); fn=int(cm.loc[0,1]); tn=int(cm.loc[0,0])
metrics=[
    ('Sensitivity (share of repricing intervals with a new date)',tp,tp+fn),
    ('Specificity (share of stable intervals with the same date)',tn,tn+fp),
    ('Positive predictive value (repricing given a new date)',tp,tp+fp),
    ('Negative predictive value (stability given the same date)',tn,tn+fn),
    ('Overall accuracy',tp+tn,tp+tn+fp+fn),
]
rows=[]
for name,num,den in metrics:
    lo,hi=proportion_confint(num,den,alpha=.05,method='wilson')
    rows.append({'metric':name,'numerator':num,'denominator':den,'estimate':num/den,'ci95_low':lo,'ci95_high':hi})
pd.DataFrame(rows).to_csv(OUT/'signal_metrics_with_ci.csv',index=False)

# Full-universe manual source audit summary.
counts=AUDIT.audit_status.value_counts()
audit_rows=[]
for status,n in counts.items(): audit_rows.append({'audit_status':status,'n':int(n)})
pd.DataFrame(audit_rows).to_csv(OUT/'manual_audit_summary.csv',index=False)
# clean categories
false_same = int(counts.get('parser_or_row_alignment_false_positive',0)+counts.get('gpl_date_parse_error_not_silent_after_audit',0))
insuff = int(counts.get('source_change_retained_but_primary_pair_insufficient_after_audit',0))
source_conf = int(counts.get('source_confirmed_same_date_repricing',0)+counts.get('source_confirmed_same_date_repricing_with_partial_parser_correction',0))
web = int(counts.get('website_derived_price_change_same_reported_gpl_date',0))
pd.DataFrame([{
    'initial_flags_audited':len(AUDIT),'not_true_same_date_repricing':false_same,
    'false_flag_rate':false_same/len(AUDIT),'real_change_but_dropped_primary_threshold':insuff,
    'source_confirmed_survey_pdf_cases':source_conf,'website_derived_retained_cases':web,
    'final_clean_same_date_repricing_pairs':int(((final.formal_revision_clean==0)&(final.any_headline_change_clean==1)).sum())
}]).to_csv(OUT/'manual_audit_key_numbers.csv',index=False)

# Brand concentration in strict synchronization sample and leave-one-brand-out table.
strict_mask=(SYNC.any_pair_active==1)&(SYNC.prior_price_vector_different==1)
sb=SYNC[strict_mask & SYNC.same_brand].copy()
brand_contrib=(sb.groupby('brand_a').agg(n_pair_years=('pair_id','size'),years=('survey_year','nunique'),agreement_rate=('exact_component_change_vector','mean')).reset_index().rename(columns={'brand_a':'brand'}))
brand_contrib.to_csv(OUT/'strict_sync_brand_contributions.csv',index=False)

# Recreate leave-one-brand-out from supplied phase3d output if available.
loo_path=ROOT/'phase3d_output'/'phase3d_leave_one_brand_out.csv'
loo=pd.read_csv(loo_path)
loo.to_csv(OUT/'leave_one_brand_out_full.csv',index=False)

print('Revision analysis complete.')
print(pd.read_csv(OUT/'attrition_gap_bounds.csv').to_string(index=False))
print(pd.read_csv(OUT/'manual_audit_key_numbers.csv').to_string(index=False))
print(brand_contrib.to_string(index=False))
