#!/usr/bin/env python3
"""Apply logged source-verification corrections and rebuild price-change fields.
The prior numeric panel is preserved; source PDFs may have documented personal-identifier redactions. The headline definition
is unchanged; no values missing in the prior panel are filled in this revision.
"""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'/'analytic_panel_pre_adversarial.csv'
d=pd.read_csv(INPUT).sort_values(['phone7','survey_year']).reset_index(drop=True)
corr=pd.read_csv(ROOT/'inputs'/'adversarial_cell_corrections.csv')
for r in corr.itertuples(index=False):
 mask=d.phone7.eq(r.phone7)&d.survey_year.eq(r.survey_year)
 assert mask.sum()==1, (r.phone7,r.survey_year)
 assert np.isclose(d.loc[mask,r.field].iloc[0],r.old_value), r
 d.loc[mask,r.field]=r.new_value
 d.loc[mask,'adversarial_correction_note']='Gabriels 2014: three prices corrected to own row; previous values matched neighboring Fuller-Sheffield row. See correction log.'
# Rebuild lags from corrected levels, never from saved delta fields.
for c in ['direct_cremation','immediate_burial','ndf']:
 prev=d.groupby('phone7')[c].shift(1)
 delta=d[c]-prev
 comparable=d.gap_years_clean.eq(1)&prev.notna()&d[c].notna()
 flag=np.where(comparable,(delta.abs()>5).astype(float),np.nan)
 d[f'prev_{c}_clean']=prev
 d[f'delta_{c}_clean']=delta
 d[f'price_change_{c}_clean']=flag
 # Refresh legacy convenience price-change fields as well.
 for field,val in [(f'prev_{c}',prev),(f'dollar_change_{c}',delta),(f'price_change_{c}',flag),(f'headline_change_{c}',flag)]:
  if field in d: d[field]=val
flags=d[[f'price_change_{c}_clean' for c in ['direct_cremation','immediate_burial','ndf']]]
d['n_headline_comparable_clean']=flags.notna().sum(axis=1)
d['n_headline_changed_clean']=flags.sum(axis=1)
d['any_headline_change_clean']=np.where(d.n_headline_comparable_clean.gt(0),d.n_headline_changed_clean.gt(0).astype(float),np.nan)
d['n_headline_comparable']=d.n_headline_comparable_clean
d['any_headline_price_change']=d.any_headline_change_clean
for c in ['funeral_no_casket','embalming']:
 prev=d.groupby('phone7')[c].shift(1);delta=d[c]-prev
 d[f'prev_{c}']=prev;d[f'dollar_change_{c}']=delta
 d[f'price_change_{c}']=np.where(d.gap_years_clean.eq(1)&d[c].notna()&prev.notna(),delta.abs().gt(5).astype(float),np.nan)
core=d[[f'price_change_{c}' for c in ['direct_cremation','immediate_burial','ndf','funeral_no_casket','embalming']]]
d['n_comparable_components']=core.notna().sum(axis=1); d['n_changed_components']=core.sum(axis=1)
d['any_core_price_change']=np.where(d.n_comparable_components.gt(0),d.n_changed_components.gt(0).astype(float),np.nan)
d['share_components_changed']=d.n_changed_components/d.n_comparable_components.replace(0,np.nan)
d.to_csv(ROOT/'analytic_panel.csv',index=False)
m=d[d.gap_years_clean.eq(1)&d.n_headline_comparable_clean.ge(2)&d.formal_revision_clean.notna()&d.any_headline_change_clean.notna()]
assert len(m)==348
assert tuple(m.groupby('formal_revision_clean').any_headline_change_clean.sum())==(40,197)
print('Source correction applied; main outcome unchanged: 40/122 and 197/226.')
