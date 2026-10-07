#!/usr/bin/env python3
"""Word count for the Economics Letters manuscript, computed two ways.
(1) Rendered-PDF count: text as an editor would see it, everything before 'References',
    minus the title block, running footer and page numbers; split by component.
(2) Source count: LaTeX with commands stripped, citations rendered as 'Author (Year)' tokens.
Economics Letters: 'Contributions are usually limited to 2,000 words excluding references.'"""
import re, subprocess, sys, json
from pathlib import Path
M = Path('manuscript')
txt = subprocess.run(['pdftotext', '-layout', str(M/'el.pdf'), '-'], capture_output=True, text=True).stdout
body = re.split(r'(?m)^\x0c?References\s*$', txt)[0]
lines = [l for l in body.splitlines()
         if not re.fullmatch(r'\s*\d+\s*', l) and 'Preprint submitted to' not in l and 'Email address' not in l]
body = '\n'.join(lines)
def wc(s): return len(re.findall(r"[A-Za-z0-9$][\w$.,%/'’()\-–]*", s))
parts = {}
tabs = 0
for cap, end in [('Table 1:', 'records excluded.'), ('Table 2:', 'outcome.')]:
    a = body.index(cap); b = body.index(end, a) + len(end)
    tabs += wc(body[a:b]); body = body[:a] + body[b:]
i_abs, i_key = body.index('Abstract'), body.index('Keywords')
i_intro = body.index('1. Introduction')
i_data = body.index('Data availability')
parts['title/author block'] = wc(body[:i_abs])
parts['abstract'] = wc(body[i_abs + len('Abstract'):i_key])
parts['keywords + JEL'] = wc(body[i_key:i_intro])
parts['main text (sections 1-4, incl. equation, headings)'] = wc(body[i_intro:i_data])
parts['tables (captions, cells, notes)'] = tabs
parts['declarations (data, AI, acknowledgments)'] = wc(body[i_data:])
tot_all = sum(parts.values())
res = {'components': parts,
       'A_text_only (abstract + main text)': parts['abstract'] + parts['main text (sections 1-4, incl. equation, headings)'],
       'B_text + tables': parts['abstract'] + parts['main text (sections 1-4, incl. equation, headings)'] + parts['tables (captions, cells, notes)'],
       'C_everything except references and title block': tot_all - parts['title/author block'],
       'D_everything except references': tot_all}
# (2) source-based count via detex for cross-check
src = (M/'manuscript.tex').read_text()
src = src.split(r'\begin{document}')[1].split(r'\bibliographystyle')[0]
for t in ['table1_signal', 'table2_age']:
    src = src.replace(r'\input{tables/%s}' % t, (M/'tables'/(t + '.tex')).read_text())
src = src.replace(r'\fi', '')
src = re.sub(r'\\cite[pt]?\*?(\[[^\]]*\])*\{([^}]*)\}', lambda m: ' '.join('Author (Year)' for _ in m.group(2).split(',')), src)
src = re.sub(r'\\(title|author|ead|affiliation)\{.*?\}\n', '', src, flags=re.S)
det = subprocess.run(['detex', '-l'], input=src, capture_output=True, text=True).stdout
res['E_source_count_detex (all incl. title, excl. refs)'] = wc(det)
json.dump(res, open('output/wordcount.json', 'w'), indent=2)
print(json.dumps(res, indent=2))
