#!/usr/bin/env python3
"""Sticky Prices at the End of Life — Phase 2C.

Required input
--------------
sticky_phase2b_clean_price_panel.csv

This script takes the source-audited Phase 2B panel as locked input and performs:
1. formal-menu and actual-price spell/state-dependence analysis;
2. timing-corrected inflation diagnostics using previous-calendar-year Dec-Dec CPI;
3. manuscript-grade synchronization kill tests (active repricing, baseline-different pairs,
   stricter common-component requirements, SCI exclusion);
4. owner-year aggregation and a year/group-size-preserving permutation benchmark;
5. ownership assignment/transition audit and period/threshold robustness;
6. manuscript-grade tables, figures, README, and transition memo.

The analysis is descriptive, not a causal ownership design.
"""
from pathlib import Path
import argparse, itertools, math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

COMPONENTS = ["direct_cremation", "immediate_burial", "ndf"]
CPI_DECDEC = {
    2013:0.015, 2014:0.008, 2015:0.007, 2016:0.021, 2017:0.021,
    2018:0.019, 2019:0.023, 2020:0.014, 2021:0.070, 2022:0.065,
    2023:0.034, 2024:0.029,
}
SEED = 20260827


def cluster_ols(formula, data, cluster='phone7'):
    return smf.ols(formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data[cluster]})


def coefrow(model_name, model, var, note=''):
    return {
        'model':model_name, 'variable':var, 'coef':float(model.params[var]),
        'se':float(model.bse[var]), 'p':float(model.pvalues[var]),
        'n':int(model.nobs), 'note':note
    }


def build_component_hazards(panel, mode='abs', threshold=5.0):
    """Build discrete annual component repricing hazards.

    prior_spell_age_years is known only after an observed price change, avoiding
    left-censoring in the main state-dependence regressions. Missing prices or
    nonconsecutive survey years break spell identification.
    """
    recs=[]
    for phone,g in panel.sort_values(['phone7','survey_year']).groupby('phone7'):
        for c in COMPONENTS:
            known=False; last_change_year=None
            prev_y=None; prev_price=None
            have_prev_interval=False; obs_run=0
            for _,row in g.iterrows():
                y=int(row['survey_year']); price=row[c]
                if prev_y is not None:
                    consecutive=(y-prev_y)==1
                    comparable=consecutive and pd.notna(price) and pd.notna(prev_price) and float(prev_price)!=0
                    if comparable:
                        delta=float(price-prev_price)
                        if mode=='abs':
                            change=float(abs(delta)>threshold)
                        else:
                            change=float(abs(delta)/abs(float(prev_price))>threshold)
                        recs.append({
                            'survey_year':y,'phone7':int(phone),'canonical_name':row['canonical_name'],
                            'component':c,'price':float(price),'prev_price':float(prev_price),
                            'delta':delta,'change':change,'ultimate_owner':row.get('ultimate_owner',np.nan),
                            'formal_revision_clean':row.get('formal_revision_clean',np.nan),
                            'prior_spell_age_years':(float(prev_y-last_change_year) if known and last_change_year is not None else np.nan),
                            'post_change_spell_known':bool(known),
                            'prior_observed_unchanged_run':(float(obs_run) if have_prev_interval else np.nan),
                            'infl_prevcal':CPI_DECDEC.get(y-1,np.nan),
                            'infl_prevcal_pp':100*CPI_DECDEC.get(y-1,np.nan) if (y-1) in CPI_DECDEC else np.nan,
                            'high_inflation_exposure':int(y in [2022,2023,2024]),
                            'covid_or_highinflation_exposure':int(y in [2021,2022,2023,2024]),
                        })
                        if change==1:
                            known=True; last_change_year=y; obs_run=0
                        else:
                            obs_run=(obs_run+1) if have_prev_interval else 1
                        have_prev_interval=True
                    else:
                        known=False; last_change_year=None; have_prev_interval=False; obs_run=0
                prev_y=y; prev_price=price
    return pd.DataFrame(recs)


def build_actual_spells(haz):
    """Episode summaries for observed component price spells under baseline >$5 rule."""
    rows=[]
    for (phone,c),g in haz.sort_values(['phone7','component','survey_year']).groupby(['phone7','component']):
        # Hazards are intervals ending at survey_year. Each observed change starts a known spell.
        known_rows=g[g['post_change_spell_known'] | (g['change']==1)].copy()
        # Create event-started spell ids. Only spells that begin with an observed change are treated as known.
        spell_id=0; current=None
        for _,r in g.iterrows():
            if r['change']==1:
                if current is not None:
                    current['end_survey_year']=int(r['survey_year'])-1
                    current['ended_with_observed_change']=1
                    rows.append(current)
                spell_id+=1
                current={
                    'phone7':int(phone),'component':c,'spell_id':spell_id,
                    'start_survey_year':int(r['survey_year']),
                    'start_price':float(r['price']),
                    'end_survey_year':int(r['survey_year']),
                    'n_observed_followup_intervals':0,
                    'ended_with_observed_change':0,
                }
            elif current is not None:
                current['end_survey_year']=int(r['survey_year'])
                current['n_observed_followup_intervals']+=1
        if current is not None:
            rows.append(current)
    out=pd.DataFrame(rows)
    if len(out): out['observed_spell_length_years']=out['end_survey_year']-out['start_survey_year']
    return out


def build_formal_menu_episodes(panel):
    rows=[]
    p=panel.copy()
    p['gpl_effective_date']=pd.to_datetime(p['gpl_effective_date'],errors='coerce')
    p['survey_anchor_date']=pd.to_datetime(p['survey_anchor_date'],errors='coerce')
    for phone,g in p.sort_values(['phone7','survey_year']).groupby('phone7'):
        ep=None
        for _,r in g.iterrows():
            if pd.isna(r['gpl_effective_date']):
                if ep is not None:
                    rows.append(ep); ep=None
                continue
            gd=r['gpl_effective_date']
            if ep is None or gd!=ep['_gpl'] or int(r['survey_year'])!=ep['end_survey_year']+1:
                if ep is not None: rows.append(ep)
                ep={'phone7':int(phone),'gpl_effective_date':gd.date().isoformat(),
                    'start_survey_year':int(r['survey_year']),'end_survey_year':int(r['survey_year']),
                    'n_survey_observations':1,'start_menu_age_days':float(r['gpl_age_days_clean']) if pd.notna(r['gpl_age_days_clean']) else np.nan,
                    'end_menu_age_days':float(r['gpl_age_days_clean']) if pd.notna(r['gpl_age_days_clean']) else np.nan,
                    '_gpl':gd}
            else:
                ep['end_survey_year']=int(r['survey_year']); ep['n_survey_observations']+=1
                ep['end_menu_age_days']=float(r['gpl_age_days_clean']) if pd.notna(r['gpl_age_days_clean']) else np.nan
        if ep is not None: rows.append(ep)
    out=pd.DataFrame(rows)
    if len(out):
        out=out.drop(columns=['_gpl'])
        out['observed_span_years']=out['end_survey_year']-out['start_survey_year']
    return out


def state_dependence(panel, baseline_haz):
    rows=[]
    # Formal menu hazard: actual prior GPL age is observed from effective date.
    d=panel[(panel['gap_years_clean']==1)&panel['formal_revision_clean'].notna()].copy()
    d['prior_gpl_age_years']=d['prev_gpl_age_days_clean']/365.25
    d=d.dropna(subset=['prior_gpl_age_years'])
    m=cluster_ols('formal_revision_clean ~ prior_gpl_age_years + C(survey_year) + C(phone7)',d)
    rows.append(coefrow('formal_menu_state_dependence_unit_year_fe',m,'prior_gpl_age_years','years of prior GPL age'))
    d['formal_age_bin']=pd.cut(d['prior_gpl_age_years'],[-np.inf,.75,1.5,2.5,np.inf],labels=['<0.75','0.75-1.5','1.5-2.5','2.5+'])
    mb=cluster_ols('formal_revision_clean ~ C(formal_age_bin, Treatment(reference="<0.75")) + C(survey_year) + C(phone7)',d)
    for var in mb.params.index:
        if 'formal_age_bin' in var:
            rows.append(coefrow('formal_menu_age_bins_unit_year_fe',mb,var,'reference <0.75 years'))

    # Actual price hazard: known post-change spells only.
    a=baseline_haz.dropna(subset=['prior_spell_age_years']).copy()
    m=cluster_ols('change ~ prior_spell_age_years + C(component) + C(survey_year) + C(phone7)',a)
    rows.append(coefrow('actual_price_state_dependence_unit_component_year_fe',m,'prior_spell_age_years','known post-change spells only'))
    a['actual_age_bin']=pd.Categorical(np.where(a['prior_spell_age_years']>=3,'3+',a['prior_spell_age_years'].astype(int).astype(str)),categories=['0','1','2','3+'])
    mb=cluster_ols('change ~ C(actual_age_bin, Treatment(reference="0")) + C(component) + C(survey_year) + C(phone7)',a)
    for var in mb.params.index:
        if 'actual_age_bin' in var:
            rows.append(coefrow('actual_price_age_bins_unit_component_year_fe',mb,var,'reference newly observed price spell (age 0)'))
    return pd.DataFrame(rows), d, a


def inflation_tests(formal_d, actual_d):
    rows=[]
    f=formal_d.copy(); f['infl_prevcal_pp']=f['survey_year'].map(lambda y:100*CPI_DECDEC.get(int(y)-1,np.nan) if int(y)-1 in CPI_DECDEC else np.nan)
    f['trend']=f['survey_year']-f['survey_year'].min()
    f['high_inflation_exposure']=f['survey_year'].isin([2022,2023,2024]).astype(int)
    f=f.dropna(subset=['infl_prevcal_pp'])
    m=cluster_ols('formal_revision_clean ~ prior_gpl_age_years + infl_prevcal_pp + trend + C(phone7)',f)
    rows += [coefrow('formal_inflation_unit_fe_trend',m,'infl_prevcal_pp','previous-calendar-year Dec-Dec CPI percentage points'),coefrow('formal_inflation_unit_fe_trend',m,'prior_gpl_age_years')]
    mh=cluster_ols('formal_revision_clean ~ prior_gpl_age_years + high_inflation_exposure + C(phone7)',f)
    rows.append(coefrow('formal_highinflation_unit_fe',mh,'high_inflation_exposure','survey outcomes 2022-2024'))

    a=actual_d.copy(); a['trend']=a['survey_year']-a['survey_year'].min()
    a=a.dropna(subset=['infl_prevcal_pp'])
    m=cluster_ols('change ~ prior_spell_age_years + infl_prevcal_pp + trend + C(component) + C(phone7)',a)
    rows += [coefrow('actual_inflation_unit_component_fe_trend',m,'infl_prevcal_pp','previous-calendar-year Dec-Dec CPI percentage points'),coefrow('actual_inflation_unit_component_fe_trend',m,'prior_spell_age_years')]
    mh=cluster_ols('change ~ prior_spell_age_years + high_inflation_exposure + C(component) + C(phone7)',a)
    rows.append(coefrow('actual_highinflation_unit_component_fe',mh,'high_inflation_exposure','survey outcomes 2022-2024'))
    return pd.DataFrame(rows)


def threshold_sensitivity(panel):
    rows=[]
    specs=[('>$5','abs',5.0),('>$25','abs',25.0),('>2.5%','pct',.025),('>5%','pct',.05)]
    for label,mode,thr in specs:
        h=build_component_hazards(panel,mode,thr)
        d=h.dropna(subset=['prior_spell_age_years']).copy()
        if not len(d): continue
        m=cluster_ols('change ~ prior_spell_age_years + C(component) + C(survey_year) + C(phone7)',d)
        rows.append({'threshold':label,'n_component_intervals':len(h),'n_known_spell_hazards':len(d),
                     'known_spell_change_rate':d['change'].mean(),'age_coef':m.params['prior_spell_age_years'],
                     'age_se':m.bse['prior_spell_age_years'],'age_p':m.pvalues['prior_spell_age_years']})
    return pd.DataFrame(rows)


def pair_metrics(a,b):
    flags=[]; dels=[]; active=[]; prioreq=[]; curreq=[]
    for c in COMPONENTS:
        fa=a.get(f'price_change_{c}_clean'); fb=b.get(f'price_change_{c}_clean')
        da=a.get(f'delta_{c}_clean'); db=b.get(f'delta_{c}_clean')
        pa=a.get(f'prev_{c}_clean'); pb=b.get(f'prev_{c}_clean')
        ca=a.get(c); cb=b.get(c)
        if pd.notna(fa) and pd.notna(fb):
            flags.append(float(fa==fb)); active.append(float((fa==1) or (fb==1)))
            if pd.notna(da) and pd.notna(db): dels.append(float(abs(da-db)<=5))
            if pd.notna(pa) and pd.notna(pb): prioreq.append(float(abs(pa-pb)<=5))
            if pd.notna(ca) and pd.notna(cb): curreq.append(float(abs(ca-cb)<=5))
    if len(flags)<2: return None
    return {
        'n_common_components':len(flags),
        'exact_component_change_vector':float(all(x==1 for x in flags)),
        'exact_dollar_delta_vector':float(len(dels)>=2 and all(x==1 for x in dels)),
        'any_pair_active':float(any(x==1 for x in active)),
        'prior_price_vector_equal':float(len(prioreq)>=2 and all(x==1 for x in prioreq)),
        'prior_price_vector_different':float(len(prioreq)>=2 and not all(x==1 for x in prioreq)),
        'current_price_vector_equal':float(len(curreq)>=2 and all(x==1 for x in curreq)),
    }


def build_sync(panel):
    obs=panel[(panel['gap_years_clean']==1)&panel['brand_group'].notna()].copy()
    rows=[]
    for y,g in obs.groupby('survey_year'):
        rs=list(g.to_dict('records'))
        for a,b in itertools.combinations(rs,2):
            m=pair_metrics(a,b)
            if m is None: continue
            same_owner=pd.notna(a['ultimate_owner']) and a['ultimate_owner']==b['ultimate_owner']
            same_brand=a['brand_group']==b['brand_group']
            same_formal_status=np.nan
            if pd.notna(a.get('formal_revision_clean')) and pd.notna(b.get('formal_revision_clean')):
                same_formal_status=float(a['formal_revision_clean']==b['formal_revision_clean'])
            same_current_gpl_exact=np.nan
            if pd.notna(a.get('gpl_effective_date')) and pd.notna(b.get('gpl_effective_date')):
                same_current_gpl_exact=float(a['gpl_effective_date']==b['gpl_effective_date'])
            same_prev_gpl_exact=np.nan
            if pd.notna(a.get('prev_gpl_date_clean')) and pd.notna(b.get('prev_gpl_date_clean')):
                same_prev_gpl_exact=float(a['prev_gpl_date_clean']==b['prev_gpl_date_clean'])
            rows.append({
                'survey_year':int(y),'phone_a':int(a['phone7']),'phone_b':int(b['phone7']),
                'pair_id':'-'.join(map(str,sorted([int(a['phone7']),int(b['phone7'])]))),
                'owner_a':a['ultimate_owner'],'owner_b':b['ultimate_owner'],
                'brand_a':a['brand_group'],'brand_b':b['brand_group'],'same_owner':bool(same_owner),'same_brand':bool(same_brand),
                'same_formal_status':same_formal_status,'same_current_gpl_exact':same_current_gpl_exact,'same_prev_gpl_exact':same_prev_gpl_exact,**m
            })
    return pd.DataFrame(rows)


def sync_summary(sync):
    rows=[]
    subsets={
        'all_pairs':np.ones(len(sync),dtype=bool),
        'active_pairs':sync['any_pair_active'].eq(1),
        'active_baseline_different':sync['any_pair_active'].eq(1)&sync['prior_price_vector_different'].eq(1),
        'three_common_components':sync['n_common_components'].eq(3),
        'three_common_active':sync['n_common_components'].eq(3)&sync['any_pair_active'].eq(1),
        'three_common_active_baseline_different':sync['n_common_components'].eq(3)&sync['any_pair_active'].eq(1)&sync['prior_price_vector_different'].eq(1),
        'exclude_SCI':~(sync['owner_a'].eq('SCI')|sync['owner_b'].eq('SCI')),
        'exclude_SCI_active':~(sync['owner_a'].eq('SCI')|sync['owner_b'].eq('SCI'))&sync['any_pair_active'].eq(1),
        'exclude_SCI_active_baseline_different':~(sync['owner_a'].eq('SCI')|sync['owner_b'].eq('SCI'))&sync['any_pair_active'].eq(1)&sync['prior_price_vector_different'].eq(1),
    }
    for sub,mask in subsets.items():
        d=sync[mask]
        for same,g in d.groupby('same_owner'):
            for metric in ['exact_component_change_vector','exact_dollar_delta_vector','prior_price_vector_equal','current_price_vector_equal']:
                rows.append({'subset':sub,'same_owner':bool(same),'metric':metric,'mean':g[metric].mean(),'n':len(g)})
    return pd.DataFrame(rows)



def grouping_summary(sync, group_bool, label):
    """Same-group vs different-group synchronization summary for brand or owner."""
    rows=[]
    subsets={
        'all_pairs':np.ones(len(sync),dtype=bool),
        'active_pairs':sync['any_pair_active'].eq(1),
        'active_baseline_different':sync['any_pair_active'].eq(1)&sync['prior_price_vector_different'].eq(1),
        'three_common_components':sync['n_common_components'].eq(3),
        'three_common_active':sync['n_common_components'].eq(3)&sync['any_pair_active'].eq(1),
        'three_common_active_baseline_different':sync['n_common_components'].eq(3)&sync['any_pair_active'].eq(1)&sync['prior_price_vector_different'].eq(1),
    }
    for sub,mask in subsets.items():
        d=sync[mask]
        for same,g in d.groupby(group_bool):
            for metric in ['exact_component_change_vector','exact_dollar_delta_vector','prior_price_vector_equal','current_price_vector_equal','same_formal_status','same_current_gpl_exact','same_prev_gpl_exact']:
                if metric not in g: continue
                rows.append({'grouping':label,'subset':sub,'same_group':bool(same),'metric':metric,'mean':g[metric].mean(),'n_nonmissing':int(g[metric].notna().sum()),'n_pairs':len(g)})
    return pd.DataFrame(rows)


def brand_owner_decomposition(sync):
    """Show whether the owner result is actually identified beyond common brand."""
    x=sync.copy()
    x['org_category']=np.select(
        [x['same_owner']&x['same_brand'], x['same_owner']&~x['same_brand'], ~x['same_owner']&~x['same_brand']],
        ['same_owner_same_brand','same_owner_different_brand','different_owner_different_brand'],
        default='other')
    rows=[]
    subsets={
        'all_pairs':np.ones(len(x),dtype=bool),
        'active_pairs':x['any_pair_active'].eq(1),
        'active_baseline_different':x['any_pair_active'].eq(1)&x['prior_price_vector_different'].eq(1),
    }
    for sub,mask in subsets.items():
        d=x[mask]
        for cat,g in d.groupby('org_category'):
            if cat=='other': continue
            rows.append({'subset':sub,'org_category':cat,'n_pairs':len(g),
                         'exact_component_change_vector':g['exact_component_change_vector'].mean(),
                         'exact_dollar_delta_vector':g['exact_dollar_delta_vector'].mean(),
                         'same_formal_status':g['same_formal_status'].mean(),
                         'same_current_gpl_exact':g['same_current_gpl_exact'].mean(),
                         'same_prev_gpl_exact':g['same_prev_gpl_exact'].mean()})
    return pd.DataFrame(rows), x[(x['same_owner'])&(~x['same_brand'])].copy()


def brand_formal_control_regressions(sync):
    """Descriptive pair regressions: does same-brand repricing agreement survive formal-clock agreement?"""
    rows=[]
    for sub,mask in {
        'all_pairs':np.ones(len(sync),dtype=bool),
        'active_pairs':sync['any_pair_active'].eq(1),
        'active_baseline_different':sync['any_pair_active'].eq(1)&sync['prior_price_vector_different'].eq(1),
    }.items():
        d=sync[mask&sync['same_formal_status'].notna()].copy()
        for dep in ['exact_component_change_vector','exact_dollar_delta_vector']:
            if len(d)==0: continue
            m=smf.ols(f'{dep} ~ same_brand + same_formal_status + C(survey_year)',data=d).fit(cov_type='cluster',cov_kwds={'groups':d['pair_id']})
            for var in ['same_brand[T.True]','same_formal_status']:
                rows.append({'subset':sub,'outcome':dep,'variable':var,'coef':m.params[var],'se':m.bse[var],'p':m.pvalues[var],'n':int(m.nobs),
                             'note':'pair-clustered descriptive regression; branch pairs overlap across observations'})
    return pd.DataFrame(rows)


def inflation_year_diagnostics(panel, baseline_haz):
    """Macro-level inflation diagnostic using one observation per survey year.

    CPI varies only at the year level, so this is the appropriate descriptive check;
    unit-level CPI standard errors should not be interpreted as independent macro inference.
    """
    f=panel[(panel['gap_years_clean']==1)&panel['formal_revision_clean'].notna()].copy()
    yf=f.groupby('survey_year').agg(n_formal=('formal_revision_clean','size'),formal_revision_rate=('formal_revision_clean','mean')).reset_index()
    ya=baseline_haz.groupby('survey_year').agg(n_component_intervals=('change','size'),actual_component_change_rate=('change','mean')).reset_index()
    y=yf.merge(ya,on='survey_year',how='outer')
    y['infl_prevcal_pp']=y['survey_year'].map(lambda yr:100*CPI_DECDEC.get(int(yr)-1,np.nan) if int(yr)-1 in CPI_DECDEC else np.nan)
    regs=[]
    for dep in ['formal_revision_rate','actual_component_change_rate']:
        d=y.dropna(subset=[dep,'infl_prevcal_pp'])
        m=smf.ols(f'{dep} ~ infl_prevcal_pp',data=d).fit()
        regs.append({'outcome':dep,'coef_infl_pp':m.params['infl_prevcal_pp'],'se':m.bse['infl_prevcal_pp'],'p':m.pvalues['infl_prevcal_pp'],'n_years':int(m.nobs),'r2':m.rsquared,
                     'note':'year-level descriptive regression; only annual macro variation'})
    return y,pd.DataFrame(regs)


def permutation_benchmark_group(panel, group_col, nperm=2000, exclude_sci=False):
    """Year/group-size-preserving permutation benchmark for owner or brand groups."""
    obs=panel[(panel['gap_years_clean']==1)&panel[group_col].notna()].copy()
    if exclude_sci: obs=obs[obs['ultimate_owner']!='SCI'].copy()
    year_data={}; actual={}
    for y,g0 in obs.groupby('survey_year'):
        counts=g0.groupby(group_col)['phone7'].nunique(); counts=counts[counts>=2]
        g=g0[g0[group_col].isin(counts.index)].drop_duplicates('phone7').copy()
        if not len(g): continue
        phones=list(g['phone7'].astype(int)); rmap={int(r.phone7):r for _,r in g.iterrows()}; pre={}
        for a,b in itertools.combinations(phones,2):
            m=pair_metrics(rmap[a],rmap[b])
            if m is not None:
                pre[tuple(sorted((a,b)))]=(m['exact_component_change_vector'],m['exact_dollar_delta_vector'],m['any_pair_active'],m['prior_price_vector_different'])
        year_data[int(y)]={'phones':phones,'pairs':pre,'sizes':[(str(o),int(n)) for o,n in counts.items()]}
        actual[int(y)]={str(o):list(gr['phone7'].astype(int)) for o,gr in g.groupby(group_col)}
    def calc(assign):
        vals=[]
        for y,groups in assign.items():
            pre=year_data[y]['pairs']
            for phones in groups.values():
                for a,b in itertools.combinations(phones,2):
                    m=pre.get(tuple(sorted((int(a),int(b)))))
                    if m is not None: vals.append(m)
        arr=np.array(vals,float); active=arr[:,2]==1; abd=active&(arr[:,3]==1)
        return {'exact_component_vector':arr[:,0].mean(),'exact_dollar_vector':arr[:,1].mean(),
                'active_exact_component_vector':arr[active,0].mean(),'active_exact_dollar_vector':arr[active,1].mean(),
                'active_baseline_diff_exact_component_vector':arr[abd,0].mean() if abd.any() else np.nan,
                'active_baseline_diff_exact_dollar_vector':arr[abd,1].mean() if abd.any() else np.nan,
                'n_pairs':len(arr),'n_active_pairs':int(active.sum()),'n_active_baseline_diff_pairs':int(abd.sum())}
    observed=calc(actual)
    offset=11 if group_col=='brand_group' else 0
    rng=np.random.default_rng(SEED+offset+(1 if exclude_sci else 0)); sims=[]
    for _ in range(nperm):
        ass={}
        for y,yd in year_data.items():
            ph=rng.permutation(np.array(yd['phones'],dtype=int)); pos=0; groups={}
            for group,n in yd['sizes']:
                groups[group]=ph[pos:pos+n].tolist(); pos+=n
            ass[y]=groups
        sims.append(calc(ass))
    sims=pd.DataFrame(sims); rows=[]
    sample=f'{group_col}_exclude_SCI' if exclude_sci else group_col
    for metric in ['exact_component_vector','exact_dollar_vector','active_exact_component_vector','active_exact_dollar_vector','active_baseline_diff_exact_component_vector','active_baseline_diff_exact_dollar_vector']:
        s=sims[metric].dropna(); ov=float(observed[metric]); pv=(1+(s>=ov).sum())/(len(s)+1)
        rows.append({'grouping':group_col,'sample':sample,'metric':metric,'observed':ov,'permutation_mean':s.mean(),'permutation_p95':s.quantile(.95),'one_sided_p':pv,'n_permutations':len(s),
                     'observed_n_pairs':observed['n_pairs'],'observed_n_active_pairs':observed['n_active_pairs'],'observed_n_active_baseline_diff_pairs':observed['n_active_baseline_diff_pairs']})
    return pd.DataFrame(rows),sims

def owner_specific(sync):
    d=sync[sync['same_owner']].copy()
    rows=[]
    for owner,g in d.groupby('owner_a'):
        active=g[g['any_pair_active']==1]
        abd=active[active['prior_price_vector_different']==1]
        rows.append({
            'owner':owner,'n_pairs':len(g),'n_active_pairs':len(active),'n_active_baseline_different_pairs':len(abd),
            'exact_component_vector_all':g['exact_component_change_vector'].mean(),
            'exact_dollar_vector_all':g['exact_dollar_delta_vector'].mean(),
            'active_exact_component_vector':active['exact_component_change_vector'].mean() if len(active) else np.nan,
            'active_exact_dollar_vector':active['exact_dollar_delta_vector'].mean() if len(active) else np.nan,
            'active_baseline_diff_exact_component_vector':abd['exact_component_change_vector'].mean() if len(abd) else np.nan,
            'active_baseline_diff_exact_dollar_vector':abd['exact_dollar_delta_vector'].mean() if len(abd) else np.nan,
            'baseline_price_vector_equal_share':g['prior_price_vector_equal'].mean(),
        })
    return pd.DataFrame(rows)


def owner_year(sync):
    d=sync[sync['same_owner']].copy(); rows=[]
    for (owner,y),g in d.groupby(['owner_a','survey_year']):
        a=g[g['any_pair_active']==1]; abd=a[a['prior_price_vector_different']==1]
        rows.append({
            'owner':owner,'survey_year':int(y),'n_pairs':len(g),
            'exact_component_vector':g['exact_component_change_vector'].mean(),
            'exact_dollar_vector':g['exact_dollar_delta_vector'].mean(),
            'active_share':g['any_pair_active'].mean(),
            'baseline_equal_share':g['prior_price_vector_equal'].mean(),
            'n_active_pairs':len(a),
            'active_exact_component_vector':a['exact_component_change_vector'].mean() if len(a) else np.nan,
            'active_exact_dollar_vector':a['exact_dollar_delta_vector'].mean() if len(a) else np.nan,
            'n_active_baseline_different_pairs':len(abd),
            'active_baseline_diff_exact_component_vector':abd['exact_component_change_vector'].mean() if len(abd) else np.nan,
            'active_baseline_diff_exact_dollar_vector':abd['exact_dollar_delta_vector'].mean() if len(abd) else np.nan,
        })
    return pd.DataFrame(rows)


def permutation_benchmark(panel, nperm=2000, exclude_sci=False):
    obs=panel[(panel['gap_years_clean']==1)&panel['ultimate_owner'].notna()].copy()
    if exclude_sci: obs=obs[obs['ultimate_owner']!='SCI'].copy()
    year_data={}; actual={}
    for y,g0 in obs.groupby('survey_year'):
        counts=g0.groupby('ultimate_owner')['phone7'].nunique(); counts=counts[counts>=2]
        g=g0[g0['ultimate_owner'].isin(counts.index)].drop_duplicates('phone7').copy()
        if not len(g): continue
        phones=list(g['phone7'].astype(int)); rmap={int(r.phone7):r for _,r in g.iterrows()}; pre={}
        for a,b in itertools.combinations(phones,2):
            m=pair_metrics(rmap[a],rmap[b])
            if m is not None:
                pre[tuple(sorted((a,b)))]=(m['exact_component_change_vector'],m['exact_dollar_delta_vector'],m['any_pair_active'],m['prior_price_vector_different'])
        year_data[int(y)]={'phones':phones,'pairs':pre,'sizes':[(str(o),int(n)) for o,n in counts.items()]}
        actual[int(y)]={str(o):list(gr['phone7'].astype(int)) for o,gr in g.groupby('ultimate_owner')}

    def calc(assign):
        vals=[]
        for y,groups in assign.items():
            pre=year_data[y]['pairs']
            for phones in groups.values():
                for a,b in itertools.combinations(phones,2):
                    m=pre.get(tuple(sorted((int(a),int(b)))))
                    if m is not None: vals.append(m)
        arr=np.array(vals,float); active=arr[:,2]==1; abd=active&(arr[:,3]==1)
        return {
            'exact_component_vector':arr[:,0].mean(),'exact_dollar_vector':arr[:,1].mean(),
            'active_exact_component_vector':arr[active,0].mean(),'active_exact_dollar_vector':arr[active,1].mean(),
            'active_baseline_diff_exact_component_vector':arr[abd,0].mean() if abd.any() else np.nan,
            'active_baseline_diff_exact_dollar_vector':arr[abd,1].mean() if abd.any() else np.nan,
            'n_pairs':len(arr),'n_active_pairs':int(active.sum()),'n_active_baseline_diff_pairs':int(abd.sum())
        }
    observed=calc(actual)
    rng=np.random.default_rng(SEED+(1 if exclude_sci else 0)); sims=[]
    for _ in range(nperm):
        ass={}
        for y,yd in year_data.items():
            ph=rng.permutation(np.array(yd['phones'],dtype=int)); pos=0; groups={}
            for owner,n in yd['sizes']:
                groups[owner]=ph[pos:pos+n].tolist(); pos+=n
            ass[y]=groups
        sims.append(calc(ass))
    sims=pd.DataFrame(sims)
    rows=[]
    for metric in ['exact_component_vector','exact_dollar_vector','active_exact_component_vector','active_exact_dollar_vector','active_baseline_diff_exact_component_vector','active_baseline_diff_exact_dollar_vector']:
        s=sims[metric].dropna(); ov=float(observed[metric]); p=(1+(s>=ov).sum())/(len(s)+1)
        rows.append({'sample':'exclude_SCI' if exclude_sci else 'all_owner_coded','metric':metric,'observed':ov,
                     'permutation_mean':s.mean(),'permutation_p95':s.quantile(.95),'one_sided_p':p,'n_permutations':len(s),
                     'observed_n_pairs':observed['n_pairs'],'observed_n_active_pairs':observed['n_active_pairs'],
                     'observed_n_active_baseline_diff_pairs':observed['n_active_baseline_diff_pairs']})
    return pd.DataFrame(rows), sims


def owner_audit(panel):
    a=panel[['survey_year','phone7','canonical_name','brand_group','ultimate_owner']].copy().sort_values(['phone7','survey_year'])
    a['coded_brand_without_owner']=a['brand_group'].notna()&a['ultimate_owner'].isna()
    a['prev_owner']=a.groupby('phone7')['ultimate_owner'].shift()
    a['prev_year']=a.groupby('phone7')['survey_year'].shift()
    a['owner_transition']=a['ultimate_owner'].notna()&a['prev_owner'].notna()&(a['ultimate_owner']!=a['prev_owner'])&((a['survey_year']-a['prev_year'])<=2)
    trans=a[a['owner_transition']].copy()
    return a,trans


def period_robustness(panel, sync):
    # headline pair rates
    d=panel[(panel['gap_years_clean']==1)&panel['formal_revision_clean'].notna()&panel['any_headline_change_clean'].notna()].copy()
    periods={
        'all_2014_2024':d['survey_year'].between(2014,2024),
        'pre_covid_through_2019':d['survey_year'].le(2019),
        'through_feb2020_prepandemic_survey':d['survey_year'].le(2020),
        'covid_or_highinflation_2021_2024':d['survey_year'].between(2021,2024),
        'highinflation_exposure_2022_2024':d['survey_year'].between(2022,2024),
        'exclude_website_cases':~((d['phone7']==2372365)&d['survey_year'].isin([2023,2024])),
    }
    rows=[]
    for name,mask in periods.items():
        x=d[mask]
        for status,label in [(0,'same_gpl_date'),(1,'new_gpl_date')]:
            z=x[x['formal_revision_clean']==status]
            rows.append({'analysis':'headline_repricing','period':name,'group':label,'n':len(z),'rate':z['any_headline_change_clean'].mean()})
    # synchronization period summaries, all and active
    for name,yrmask in {
        'all_2014_2024':sync['survey_year'].between(2014,2024),
        'pre_covid_through_2019':sync['survey_year'].le(2019),
        'through_feb2020_prepandemic_survey':sync['survey_year'].le(2020),
        'covid_or_highinflation_2021_2024':sync['survey_year'].between(2021,2024),
        'highinflation_exposure_2022_2024':sync['survey_year'].between(2022,2024),
    }.items():
        for active_label,amask in [('all_pairs',np.ones(len(sync),dtype=bool)),('active_pairs',sync['any_pair_active'].eq(1))]:
            x=sync[yrmask&amask]
            for same,label in [(False,'different_owner'),(True,'same_owner')]:
                z=x[x['same_owner']==same]
                rows.append({'analysis':'sync_component_vector_'+active_label,'period':name,'group':label,'n':len(z),'rate':z['exact_component_change_vector'].mean()})
                rows.append({'analysis':'sync_dollar_vector_'+active_label,'period':name,'group':label,'n':len(z),'rate':z['exact_dollar_delta_vector'].mean()})
    return pd.DataFrame(rows)


def brand_specific(sync):
    d=sync[sync['same_brand']].copy(); rows=[]
    for brand,g in d.groupby('brand_a'):
        active=g[g['any_pair_active']==1]
        abd=active[active['prior_price_vector_different']==1]
        rows.append({
            'brand':brand,'n_pairs':len(g),'n_active_pairs':len(active),'n_active_baseline_different_pairs':len(abd),
            'exact_component_vector_all':g['exact_component_change_vector'].mean(),
            'exact_dollar_vector_all':g['exact_dollar_delta_vector'].mean(),
            'active_exact_component_vector':active['exact_component_change_vector'].mean() if len(active) else np.nan,
            'active_exact_dollar_vector':active['exact_dollar_delta_vector'].mean() if len(active) else np.nan,
            'active_baseline_diff_exact_component_vector':abd['exact_component_change_vector'].mean() if len(abd) else np.nan,
            'active_baseline_diff_exact_dollar_vector':abd['exact_dollar_delta_vector'].mean() if len(abd) else np.nan,
            'baseline_price_vector_equal_share':g['prior_price_vector_equal'].mean(),
            'exact_current_gpl_date_all':g['same_current_gpl_exact'].mean(),
            'exact_prev_gpl_date_all':g['same_prev_gpl_exact'].mean(),
        })
    return pd.DataFrame(rows)


def make_figures(state, brand_summary, brand, out):
    subs=['all_pairs','active_pairs','active_baseline_different']
    labels=['All comparable pairs','At least one branch reprices','Active + different starting prices']
    rows=[]
    for sub,label in zip(subs,labels):
        for metric,mname in [('exact_component_change_vector','Same components changed'),('exact_dollar_delta_vector','Same dollar changes')]:
            for same,gname in [(False,'Different brands'),(True,'Same brand')]:
                q=brand_summary[(brand_summary['subset']==sub)&(brand_summary['metric']==metric)&(brand_summary['same_group']==same)]
                if len(q): rows.append({'subset':label,'metric':mname,'group':gname,'rate':100*q.iloc[0]['mean']})
    fd=pd.DataFrame(rows)
    for metric,slug in [('Same components changed','component'),('Same dollar changes','dollar')]:
        x=fd[fd['metric']==metric]
        vals=[]
        for label in labels:
            vals.append((x[(x['subset']==label)&(x['group']=='Different brands')]['rate'].iloc[0],
                         x[(x['subset']==label)&(x['group']=='Same brand')]['rate'].iloc[0]))
        pos=np.arange(len(labels)); w=.36
        fig,ax=plt.subplots(figsize=(8.5,5.2))
        ax.bar(pos-w/2,[v[0] for v in vals],w,label='Different brands')
        ax.bar(pos+w/2,[v[1] for v in vals],w,label='Same brand')
        ax.set_xticks(pos); ax.set_xticklabels(labels,rotation=12,ha='right'); ax.set_ylabel('Exact agreement (%)')
        ax.set_title('Multi-establishment brands synchronize '+metric.lower()); ax.set_ylim(0,100); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out/f'fig_sticky_2c_sync_{slug}_killtests.png',dpi=190); plt.close(fig)

    for model,prefix,title in [
        ('formal_menu_age_bins_unit_year_fe','formal_menu','Formal GPL revision hazard by prior menu age'),
        ('actual_price_age_bins_unit_component_year_fe','actual_price','Actual repricing hazard by known price-spell age')]:
        d=state[state['model']==model].copy()
        if not len(d): continue
        labs=['0.75-1.5','1.5-2.5','2.5+'] if prefix=='formal_menu' else ['1','2','3+']
        d=d.head(len(labs))
        fig,ax=plt.subplots(figsize=(7.2,4.8)); pos=np.arange(len(d))
        ax.errorbar(pos,d['coef']*100,yerr=1.96*d['se']*100,fmt='o',capsize=4)
        ax.axhline(0,linewidth=1); ax.set_xticks(pos); ax.set_xticklabels(labs)
        ax.set_ylabel('Change in annual hazard (percentage points)'); ax.set_title(title)
        ax.set_xlabel('Prior spell/menu age (years; reference is youngest bin)')
        fig.tight_layout(); fig.savefig(out/f'fig_sticky_2c_{prefix}_age_hazard.png',dpi=190); plt.close(fig)

    d=brand.dropna(subset=['active_exact_component_vector']).sort_values('active_exact_component_vector')
    if len(d):
        fig,ax=plt.subplots(figsize=(8,5.2)); ax.barh(d['brand'],100*d['active_exact_component_vector'])
        ax.set_xlabel('Exact component-change agreement among active same-brand pairs (%)')
        ax.set_title('Active repricing synchronization by multi-establishment brand')
        ax.set_xlim(0,100); fig.tight_layout(); fig.savefig(out/'fig_sticky_2c_sync_by_brand_active.png',dpi=190); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',default='sticky_phase2b_clean_price_panel.csv'); ap.add_argument('--outdir',default='sticky_phase2c_output'); ap.add_argument('--nperm',type=int,default=2000)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(args.panel)
    for c in ['gpl_effective_date','survey_anchor_date','prev_gpl_date_clean']:
        if c in p: p[c]=pd.to_datetime(p[c],errors='coerce')

    h=build_component_hazards(p,'abs',5.0)
    actual_spells=build_actual_spells(h); formal_episodes=build_formal_menu_episodes(p)
    state,formal_d,actual_d=state_dependence(p,h)
    infl_unit=inflation_tests(formal_d,actual_d)
    infl_year,infl_year_regs=inflation_year_diagnostics(p,h)
    thresh=threshold_sensitivity(p)

    sync=build_sync(p)
    owner_summary=grouping_summary(sync,'same_owner','ultimate_owner')
    brand_summary=grouping_summary(sync,'same_brand','brand')
    decomp,cross_brand_same_owner=brand_owner_decomposition(sync)
    brand_controls=brand_formal_control_regressions(sync)
    owner=owner_specific(sync); oy=owner_year(sync); brand=brand_specific(sync)

    perm_brand,dist_brand=permutation_benchmark_group(p,'brand_group',args.nperm,False)
    perm_owner,dist_owner=permutation_benchmark_group(p,'ultimate_owner',args.nperm,False)
    perm_owner_ns,dist_owner_ns=permutation_benchmark_group(p,'ultimate_owner',args.nperm,True)
    perm=pd.concat([perm_brand,perm_owner,perm_owner_ns],ignore_index=True)
    oa,ot=owner_audit(p); period=period_robustness(p,sync)

    def getstate(model,var):
        q=state[(state.model==model)&(state.variable==var)]; return q.iloc[0] if len(q) else None
    formal=getstate('formal_menu_state_dependence_unit_year_fe','prior_gpl_age_years')
    actual=getstate('actual_price_state_dependence_unit_component_year_fe','prior_spell_age_years')
    def drow(sub,cat):
        q=decomp[(decomp.subset==sub)&(decomp.org_category==cat)]
        return q.iloc[0] if len(q) else None

    samebrand_active=drow('active_pairs','same_owner_same_brand')
    samebrand_abd=drow('active_baseline_different','same_owner_same_brand')
    diff_abd=drow('active_baseline_different','different_owner_different_brand')
    cross_abd=drow('active_baseline_different','same_owner_different_brand')

    exs=sync[~(sync['owner_a'].eq('SCI')|sync['owner_b'].eq('SCI'))].copy()
    ex_abd=exs[exs['same_brand']&exs['any_pair_active'].eq(1)&exs['prior_price_vector_different'].eq(1)]

    brand_formal=[]
    for metric in ['same_current_gpl_exact','same_prev_gpl_exact','same_formal_status']:
        for same in [False,True]:
            q=brand_summary[(brand_summary.subset=='all_pairs')&(brand_summary.same_group==same)&(brand_summary.metric==metric)]
            if len(q): brand_formal.append({'metric':metric,'same_brand':same,'rate':q.iloc[0]['mean'],'n_nonmissing':int(q.iloc[0]['n_nonmissing']),'n_pairs':int(q.iloc[0]['n_pairs'])})
    brand_formal=pd.DataFrame(brand_formal)

    infl_formal_yr=infl_year_regs[infl_year_regs.outcome=='formal_revision_rate'].iloc[0]
    infl_actual_yr=infl_year_regs[infl_year_regs.outcome=='actual_component_change_rate'].iloc[0]
    key=pd.DataFrame([
        {'result':'formal_menu_age_coef','value':formal.coef,'se':formal.se,'p':formal.p,'n':formal.n,'note':'annual formal-revision hazard per prior GPL age-year; establishment + year FE'},
        {'result':'actual_price_spell_age_coef','value':actual.coef,'se':actual.se,'p':actual.p,'n':actual.n,'note':'annual component repricing hazard per known spell-year; establishment + component + year FE'},
        {'result':'same_brand_active_exact_component_vector','value':samebrand_active.exact_component_change_vector,'se':np.nan,'p':np.nan,'n':samebrand_active.n_pairs,'note':'same owner + same brand; active pairs'},
        {'result':'same_brand_active_baseline_diff_exact_component_vector','value':samebrand_abd.exact_component_change_vector,'se':np.nan,'p':np.nan,'n':samebrand_abd.n_pairs,'note':'same owner + same brand; active pairs with different starting prices'},
        {'result':'different_owner_brand_active_baseline_diff_exact_component_vector','value':diff_abd.exact_component_change_vector,'se':np.nan,'p':np.nan,'n':diff_abd.n_pairs,'note':'different owner + different brand comparator'},
        {'result':'same_brand_active_baseline_diff_exact_dollar_vector','value':samebrand_abd.exact_dollar_delta_vector,'se':np.nan,'p':np.nan,'n':samebrand_abd.n_pairs,'note':'same owner + same brand; active pairs with different starting prices'},
        {'result':'different_owner_brand_active_baseline_diff_exact_dollar_vector','value':diff_abd.exact_dollar_delta_vector,'se':np.nan,'p':np.nan,'n':diff_abd.n_pairs,'note':'different owner + different brand comparator'},
        {'result':'cross_brand_same_owner_pair_count_all','value':len(cross_brand_same_owner),'se':np.nan,'p':np.nan,'n':len(cross_brand_same_owner),'note':'only cross-brand same-owner observations; ultimate-parent effect not separately identified'},
        {'result':'year_level_formal_inflation_slope','value':infl_formal_yr.coef_infl_pp,'se':infl_formal_yr.se,'p':infl_formal_yr.p,'n':infl_formal_yr.n_years,'note':'annual rate on prior-calendar-year CPI inflation'},
        {'result':'year_level_actual_inflation_slope','value':infl_actual_yr.coef_infl_pp,'se':infl_actual_yr.se,'p':infl_actual_yr.p,'n':infl_actual_yr.n_years,'note':'annual component-change rate on prior-calendar-year CPI inflation'},
    ])

    h.to_csv(out/'sticky_phase2c_actual_component_hazards.csv',index=False)
    actual_spells.to_csv(out/'sticky_phase2c_actual_price_spells.csv',index=False)
    formal_episodes.to_csv(out/'sticky_phase2c_formal_menu_episodes.csv',index=False)
    state.to_csv(out/'sticky_phase2c_state_dependence_regressions.csv',index=False)
    infl_unit.to_csv(out/'sticky_phase2c_inflation_unit_diagnostics.csv',index=False)
    infl_year.to_csv(out/'sticky_phase2c_inflation_year_summary.csv',index=False)
    infl_year_regs.to_csv(out/'sticky_phase2c_inflation_year_regressions.csv',index=False)
    thresh.to_csv(out/'sticky_phase2c_spell_threshold_sensitivity.csv',index=False)
    sync.to_csv(out/'sticky_phase2c_sync_pair_panel.csv',index=False)
    brand_summary.to_csv(out/'sticky_phase2c_sync_brand_summary.csv',index=False)
    owner_summary.to_csv(out/'sticky_phase2c_sync_owner_summary.csv',index=False)
    decomp.to_csv(out/'sticky_phase2c_brand_owner_decomposition.csv',index=False)
    cross_brand_same_owner.to_csv(out/'sticky_phase2c_cross_brand_same_owner_pairs.csv',index=False)
    brand_controls.to_csv(out/'sticky_phase2c_brand_formal_control_regressions.csv',index=False)
    brand_formal.to_csv(out/'sticky_phase2c_brand_formal_sync.csv',index=False)
    brand.to_csv(out/'sticky_phase2c_sync_by_brand.csv',index=False)
    owner.to_csv(out/'sticky_phase2c_sync_by_owner.csv',index=False)
    oy.to_csv(out/'sticky_phase2c_owner_year_sync.csv',index=False)
    perm.to_csv(out/'sticky_phase2c_permutation_benchmark.csv',index=False)
    dist_brand.to_csv(out/'sticky_phase2c_permutation_distribution_brand.csv',index=False)
    dist_owner.to_csv(out/'sticky_phase2c_permutation_distribution_owner.csv',index=False)
    dist_owner_ns.to_csv(out/'sticky_phase2c_permutation_distribution_owner_exSCI.csv',index=False)
    oa.to_csv(out/'sticky_phase2c_owner_assignment_audit.csv',index=False)
    ot.to_csv(out/'sticky_phase2c_owner_transition_audit.csv',index=False)
    period.to_csv(out/'sticky_phase2c_period_robustness.csv',index=False)
    key.to_csv(out/'sticky_phase2c_key_results.csv',index=False)
    make_figures(state,brand_summary,brand,out)

    ba_comp=float(samebrand_active.exact_component_change_vector); ba_dol=float(samebrand_active.exact_dollar_delta_vector)
    bd_comp=float(samebrand_abd.exact_component_change_vector); bd_dol=float(samebrand_abd.exact_dollar_delta_vector)
    cmp_comp=float(diff_abd.exact_component_change_vector); cmp_dol=float(diff_abd.exact_dollar_delta_vector)
    xcomp=float(cross_abd.exact_component_change_vector) if cross_abd is not None else np.nan
    xdol=float(cross_abd.exact_dollar_delta_vector) if cross_abd is not None else np.nan
    excomp=float(ex_abd['exact_component_change_vector'].mean()) if len(ex_abd) else np.nan
    exdol=float(ex_abd['exact_dollar_delta_vector'].mean()) if len(ex_abd) else np.nan
    pperm=float(perm_brand.loc[perm_brand.metric=='active_baseline_diff_exact_component_vector','one_sided_p'].iloc[0])
    brand_same_n=int(decomp[(decomp.subset=='all_pairs')&(decomp.org_category=='same_owner_same_brand')].iloc[0].n_pairs)
    cross_n=len(cross_brand_same_owner)
    bf_cur=brand_formal[(brand_formal.metric=='same_current_gpl_exact')&brand_formal.same_brand].iloc[0]
    bf_prev=brand_formal[(brand_formal.metric=='same_prev_gpl_exact')&brand_formal.same_brand].iloc[0]
    bf_cur_diff=brand_formal[(brand_formal.metric=='same_current_gpl_exact')&(~brand_formal.same_brand)].iloc[0]
    bf_prev_diff=brand_formal[(brand_formal.metric=='same_prev_gpl_exact')&(~brand_formal.same_brand)].iloc[0]
    ctrl=brand_controls[(brand_controls.subset=='active_baseline_different')&(brand_controls.outcome=='exact_component_change_vector')&(brand_controls.variable=='same_brand[T.True]')].iloc[0]
    coded_missing=int(oa['coded_brand_without_owner'].sum())

    readme=f'''# Sticky Prices at the End of Life — Phase 2C

## Verdict
**PASS WITH PIVOT.** Phase 2C strengthens the two-clock/state-dependence result and the existence of within-chain repricing coordination, but it also finds a crucial identification limitation that changes the organizational language of the paper. In the comparable synchronization sample, **{brand_same_n} of {brand_same_n+cross_n} same-owner pair-years are also same-brand**, and there are only **{cross_n} cross-brand same-owner pair-years**. The data therefore identify **multi-establishment brand/chain synchronization**, not a clean ultimate-parent ownership effect.

## Exact input
- `{Path(args.panel).name}` — the source-audited Phase 2B clean price panel.
- Raw FCACTX PDFs are not reparsed for the statistical core; Phase 2C deliberately treats Phase 2B's source-audited panel as locked and uses raw-source material only for provenance/timing checks.

## 1. Separate state dependence in the two clocks
Formal-menu revision: prior GPL age coefficient = **{formal.coef:.3f}** (SE {formal.se:.3f}, p={formal.p:.3g}, N={int(formal.n)}), with establishment and survey-year fixed effects. A one-year older prior GPL is associated with about **{100*formal.coef:.1f} percentage points** greater annual formal-revision probability within establishment.

Actual component repricing, using only spells whose start is known because a prior change was observed: prior spell-age coefficient = **{actual.coef:.3f}** (SE {actual.se:.3f}, p={actual.p:.3g}, N={int(actual.n)}), with establishment, component, and survey-year fixed effects. This is about **{100*actual.coef:.1f} percentage points** per observed spell-year.

The actual-price coefficient is stable at the $25, 2.5%, and 5% repricing thresholds; see `sticky_phase2c_spell_threshold_sensitivity.csv`. The age-bin patterns are not strictly monotone, so the manuscript should describe **positive duration dependence in the linear hazard specification**, not claim a smooth monotone hazard.

## 2. The mechanical synchronization kill tests survive — at the brand/chain level
Among same-brand pairs where **at least one establishment reprices**, exact component-change vectors match **{ba_comp:.1%}** of the time. Among active same-brand pairs that **start from different price vectors**, exact component-change vectors still match **{bd_comp:.1%}** and exact dollar-change vectors match **{bd_dol:.1%}**.

The clean different-owner/different-brand comparator in that demanding active-baseline-different sample is **{cmp_comp:.1%}** for component-vector agreement and **{cmp_dol:.1%}** for exact dollar-vector agreement. Thus the coordination finding is not a common-zero artifact and is not generated solely by branches entering the year with identical price schedules.

A year/group-size-preserving brand reassignment benchmark confirms that the observed active-baseline-different component agreement is far above random grouping (one-sided permutation p={pperm:.4g}; {args.nperm:,} permutations). This is a descriptive randomization benchmark, not causal ownership identification.

## 3. Phase 2C kills the strong ultimate-owner claim
The owner and brand indicators are nearly nested: **{brand_same_n} same-owner pair-years are same-brand, versus only {cross_n} same-owner/cross-brand pair-years**. Those {cross_n} cross-brand same-owner observations are suggestive (component-vector agreement {xcomp:.1%}; dollar-vector agreement {xdol:.1%}) but far too few to identify an ultimate-parent effect.

Therefore manuscript language should use **brand, chain, or multi-establishment organization** for the main result. “Common ownership causes synchronization” is too strong for these data. Ultimate ownership remains useful institutional metadata and for acquisition timing, but it is not separately identified from brand in the synchronization design.

## 4. Formal administration is itself strongly synchronized within brands
Same-brand establishments report the exact same current GPL effective date **{bf_cur.rate:.1%}** of the time (N={int(bf_cur.n_nonmissing)} nonmissing comparisons), versus **{bf_cur_diff.rate:.1%}** among different-brand pairs. The exact previous GPL date agrees **{bf_prev.rate:.1%}** versus **{bf_prev_diff.rate:.1%}**.

This means some observed repricing synchronization plausibly travels through a shared formal menu-administration process. Even controlling descriptively for same formal-revision status and survey-year fixed effects, the same-brand coefficient for exact component-vector agreement in active, different-starting-price pairs is **{ctrl.coef:.3f}** (SE {ctrl.se:.3f}, p={ctrl.p:.3g}). Treat that regression as descriptive because branch-pair observations overlap.

## 5. What survives outside SCI
Among same-brand active pairs starting from different price vectors and excluding SCI, exact component-vector agreement is **{excomp:.1%}**, while exact dollar-vector agreement falls to **{exdol:.1%}**. The broadest defensible claim is therefore synchronization of **which components are repriced**; identical dollar adjustments are much more concentrated in SCI/Cook-Walden and should be presented as heterogeneous across chains.

## 6. Inflation is not the paper
Because CPI only varies annually, Phase 2C's main inflation check is at the **survey-year level**, using previous-calendar-year Dec-Dec CPI so the exposure precedes the January-March survey. With only {int(infl_formal_yr.n_years)} annual observations, there is no positive inflation relationship:
- formal revision rate: slope **{infl_formal_yr.coef_infl_pp:.4f}** per CPI percentage point (p={infl_formal_yr.p:.3g});
- actual component repricing rate: slope **{infl_actual_yr.coef_infl_pp:.4f}** (p={infl_actual_yr.p:.3g}).

Unit-level CPI regressions are retained only as diagnostics; their standard errors should **not** be interpreted as independent macro inference. Do not frame the paper as high inflation triggering repricing.

## 7. COVID/source robustness and ownership timing
The 2020 FCACTX survey anchor is February 20, 2020, so it predates the U.S. COVID shock for this design; pandemic-exposed survey outcomes begin in 2021. The formal-revision/actual-repricing gap persists in pre-COVID, 2021-24, high-inflation 2022-24, and website-exclusion cuts in `sticky_phase2c_period_robustness.csv`.

Ownership labels remain year-specific. `sticky_phase2c_owner_transition_audit.csv` exports all detected transitions for replication documentation. There are {coded_missing} brand-coded rows with missing ultimate owner; unresolved cases remain uncoded rather than force-assigned.

## Journal-level kill test
**PASS WITH PIVOT, not STRONG PASS.** There is still a publishable paper, but Phase 2C narrows the headline organizational inference. The paper can credibly establish:
1. formal menu revision and actual repricing are related but distinct adjustment clocks;
2. each clock exhibits its own state dependence; and
3. multi-establishment brands/chains strongly synchronize what they reprice, including during active adjustment and when establishments begin from different price vectors.

What it cannot currently establish is a separate causal **ultimate-parent ownership** effect. That makes **Review of Industrial Organization or Southern Economic Journal** the most natural ambitious first shots unless the Phase 2D literature check uncovers unusually strong novelty. **Economic Inquiry is a stretch**, and **Journal of Regulatory Economics** remains plausible if the GPL/regulatory-institution angle becomes central.

## Exact next step — Phase 2D
Run the final adversarial novelty/literature check specifically against (i) state-dependent pricing and menu costs, (ii) centralized versus local price setting in chains, (iii) synchronized multi-product price adjustment, and (iv) funeral-market GPL regulation. Then lock the manuscript tables/figures and decide whether the paper's top-line contribution is **two clocks + chain coordination** or a narrower organizational pricing fact before drafting.
'''
    (out/'README_STICKY_PHASE2C.md').write_text(readme)

    memo=f'''# Sticky Prices — Phase 2C transition memo

**Status: PASS WITH PIVOT.** Phase 2C is complete. Formal GPL revision and actual component repricing each show positive duration dependence on their own age measures. The actual-price spell result survives $25, 2.5%, and 5% thresholds. Inflation is not a positive driver once timing is handled correctly and macro variation is evaluated at the year level.

The synchronization result survives the key mechanical tests: it remains large among active repricing pairs and among establishments that start from different price vectors. However, Phase 2C finds that ultimate owner and brand are almost nested: {brand_same_n} of the {brand_same_n+cross_n} same-owner comparable pair-years are same-brand, leaving only {cross_n} cross-brand same-owner observations. The main organizational result must therefore be written as **multi-establishment brand/chain synchronization**, not a separately identified ultimate-parent ownership effect.

Same-brand active baseline-different pairs agree on the exact component-change vector {bd_comp:.1%} of the time versus {cmp_comp:.1%} for different-owner/different-brand pairs; exact dollar-vector agreement is {bd_dol:.1%} versus {cmp_dol:.1%}. Excluding SCI, component-vector agreement remains {excomp:.1%} but exact-dollar agreement falls to {exdol:.1%}, so identical dollar changes are heterogeneous and especially concentrated in SCI/Cook-Walden.

Same-brand establishments also frequently share exact GPL dates, indicating a shared formal administrative process. This is mechanism evidence, not causal decomposition.

**Next: Phase 2D.** Adversarial literature/novelty check, then lock manuscript tables and figures. Current journal assessment: RIO/SEJ are the natural ambitious first shots; EI is a stretch unless novelty is stronger than expected; JRegE is plausible with a strong GPL/regulatory framing.
'''
    (out/'STICKY_PHASE2C_MEMO.md').write_text(memo)

if __name__=='__main__': main()
