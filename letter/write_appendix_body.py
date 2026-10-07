#!/usr/bin/env python3
"""Write manuscript/appendix_body.tex (online appendix text) from output/appendix.json."""
import json
r = json.load(open('output/appendix.json'))
A1 = r['A1_by_measure']; per = r['by_period']; A2 = r['A2_noise']
fee = A1['Basic-services fee, annual']; dc = A1['Direct cremation, annual']; ib = A1['Immediate burial incl. casket, annual']
pd_ = A2['persistent_drop_unverifiable']
T = open('appendix_template.tex').read()
vals = {
 'FEE_SENS': f"{fee['sens']:.3f}", 'FEE_SAME': f"{fee['same_chg']} of {fee['same_n']}", 'FEE_SPEC': f"{fee['spec']:.3f}",
 'DC_SAME': f"{dc['same_chg']} of {dc['same_n']}", 'IB_SAME': f"{ib['same_chg']} of {ib['same_n']}",
 'J_EARLY': f"{per['2014--2019']['J']:.3f}", 'J_LATE': f"{per['2020--2024']['J']:.3f}",
 'PD_RATES': f"{pd_['new_chg']}/{pd_['new_n']} ({100*pd_['rate_new']:.1f}\\%) and {pd_['same_chg']}/{pd_['same_n']} ({100*pd_['rate_same']:.1f}\\%)",
 'PD_J': f"{pd_['J']:.3f}", 'SD_EST': str(A2['same_date_repricing_establishments']),
 'SD_MAX': str(round(100 * A2['same_date_repricing_max_share_one_establishment'])),
 'WORST': f"{r['worst_case_gap']:.3f}",
 'GAP_MIN': f"{min([x['gap'] for x in r['A3_robustness']] + r['leave_one_brand_out_gap_range'] + r['leave_one_establishment_out_gap_range']):.3f}",
 'GAP_MAX': f"{max([x['gap'] for x in r['A3_robustness']] + r['leave_one_brand_out_gap_range'] + r['leave_one_establishment_out_gap_range']):.3f}",
}
for k, v in vals.items(): T = T.replace('<<' + k + '>>', v)
assert '<<' not in T
open('manuscript/appendix_body.tex', 'w').write(T)
print('appendix_body.tex written')
