#!/usr/bin/env python3
"""Build the two Economics Letters exhibits from el_analysis.py outputs (and wa_output if present)."""
import json
from pathlib import Path
import pandas as pd
O = Path('output'); T = Path('manuscript/tables'); T.mkdir(parents=True, exist_ok=True)
r = json.load(open(O/'el_results.json'))
kc = pd.read_csv(O/'kc_signal_metrics.csv')
age = pd.read_csv(O/'list_age_tests.csv').set_index('model')

def row(label, gap, n, nr, nn, sr, sn, se, sp, J):
    return (f"{label} & {gap} & {n} & {nr}/{nn} ({100*nr/nn:.1f}) & {sr}/{sn} ({100*sr/sn:.1f}) & "
            f"{se:.3f} & {sp:.3f} & {J:.3f} \\\\")
A = r; B = r['two_year']['all_two_year_links']
C = json.load(open(O/'appendix.json'))['A1_by_measure']['Direct cremation, two-year links']
rows = [row('Austin, any of three', '1', A['n'], A['new_date_reprice'], A['new_date_n'], A['same_date_reprice'], A['same_date_n'],
            A['sensitivity'], A['specificity'], A['J']),
        row('Austin, any of three', '2', B['n'], B['new_reprice'], B['new_n'], B['same_reprice'], B['same_n'],
            B['sensitivity'], B['specificity'], B['J']),
        row('Austin, cremation', '2', C['n'], C['new_chg'], C['new_n'], C['same_chg'], C['same_n'],
            C['sens'], C['spec'], C['J'])]
for w, g in [('2018--2020', '2'), ('2020--2023', '3')]:
    k = kc[(kc.window == w.replace('--', '-')) & (kc.outcome == 'direct_cremation')].iloc[0]
    rows.append(row(f'Kansas City, cremation, {w[:4]}--{w[-2:]}$^{{a}}$', g, int(k.n), int(k.new_reprice), int(k.new_n), int(k.same_reprice),
                    int(k.same_n), k.sensitivity, k.specificity, k.J))
t1 = r"""\begin{table}[t]
\centering\footnotesize
\caption{Effective-date changes and observed repricing.}\label{tab:signal}
\setlength{\tabcolsep}{2.5pt}\scriptsize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
 & Gap & & \multicolumn{2}{c}{Repriced/intervals (\%)} & & & \\
\cmidrule(lr){4-5}
Market, price measure & (yrs) & $N$ & New date & Same date & Sens. & Spec. & $J$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{minipage}{\textwidth}\vspace{3pt}\scriptsize
Notes: Repricing: a comparable price changes by more than \$5. Sens.: share of repricing intervals with a new date; Spec.: share of stable intervals with an unchanged date; $J$ = Sens. + Spec. $-$ 1. ``Any of three'': direct cremation, immediate burial incl.~casket, or basic-services fee; at least two comparable. Cremation rows use all links at which direct cremation is comparable. $^{a}$Month-level dates from published survey tables; carried-forward records excluded.
\end{minipage}
\end{table}
"""
(T/'table1_signal.tex').write_text(t1)

def cell(k):
    x = age.loc[k]
    c = f"{x.coef:.3f}".replace('-', '$-$')
    return c, f"({x.se:.3f})", f"[{x.wild_p:.3f}]"
specs = [('(1) date change on prior age', '(2) price change on prior age', '(4) date change on age, given price', '(3) price change on age, given date'),
         ('(1y) date change on prior age, year FE', '(2y) price change on prior age, year FE', '(4y) date change on age given price, year FE', '(3y) price change on age given date, year FE')]
lines = []
for lab, ks in zip(['Establishment and year effects', 'Year effects only'], specs):
    c = [cell(k) for k in ks]
    lines.append(r'\multicolumn{5}{@{}l}{\emph{' + lab + r'}}\\')
    lines.append('Prior list age (years) & ' + ' & '.join(x[0] for x in c) + r' \\')
    lines.append(' & ' + ' & '.join(x[1] for x in c) + r' \\')
    lines.append(' & ' + ' & '.join(x[2] for x in c) + r' \\[2pt]')
t2 = r"""\begin{table}[t]
\centering\footnotesize
\caption{Prior price-list age and the two measures of repricing, Greater Austin.}\label{tab:age}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
Outcome: & Date change & Price change & Date change & Price change \\
Also controls for: & -- & -- & Price change & Date change \\
 & (1) & (2) & (3) & (4) \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular*}
\begin{minipage}{\textwidth}\vspace{3pt}\scriptsize
Notes: Linear probability models; 348 annual intervals, 53 establishments. Prior list age is the age of the effective date on the preceding survey's list. Standard errors clustered by establishment in parentheses; wild-cluster bootstrap $p$-values (Rademacher, 4,999 draws) in brackets. Under nondifferential date error the coefficient in column (3) is zero and column (1) equals $J\times$ column (2). Column (4) conditions on an outcome.
\end{minipage}
\end{table}
"""
(T/'table2_age.tex').write_text(t2)
print(t1); print(t2)
