# When Disclosure Dates and Prices Diverge: Evidence from the Funeral Rule

Replication package for

> Canann, Taylor J. (2026). *When Disclosure Dates and Prices Diverge: Evidence from the Funeral Rule.* Working paper, SSRN 7548560. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7548560>

The paper tests whether the effective date printed on a funeral provider's General Price List (GPL), which the FTC Funeral Rule requires, records when the provider's prices change. It uses 348 audited annual intervals of Greater Austin price surveys (2013–2024), with Kansas City as a second market.

One command reproduces every table, every appendix table, and every number in the text, and checks each number in the manuscript against the computed value.

## Quick start

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

The run takes about 1–2 minutes on a laptop and ends with `57/57 checks pass`. Tested with Python 3.11 and the pinned packages in `requirements.txt`. Random procedures (wild-cluster bootstrap, block bootstrap, simulation) use fixed seeds.

The PDFs (`letter/manuscript/el.pdf`, `ssrn.pdf`, `appendix.pdf`) are built only if `latexmk` is installed, with TeX Live including the Elsevier `elsarticle` class. Without LaTeX, all tables are still written to `letter/manuscript/tables/` and all numbers are still checked.

## What is in the repository

```
run_all.sh                 one-command reproduction (Austin pipeline -> letter analysis -> PDFs -> number checks)
requirements.txt           pinned Python packages
austin_pipeline/           Greater Austin data and the scripts that build the audited panel and base outputs
  raw_fcactx_surveys/      13 FCACTX survey publications, 2013-2024 and 2026 (no 2025 survey)
  inputs/                  audited establishment-year panel (pre-correction), logged cell corrections,
                           visual re-verification sample, source manifests with SHA-256 hashes
  source_results/          audit ledger of every interval initially flagged as same-date repricing
  apply_source_corrections.py, replicate_CEP*.py, phase3d_analysis.py, analysis.py,
  revision_analysis.py, adversarial_analysis.py
kansas_city/               Kansas City aggregate files (FCA of Greater Kansas City 2018, 2020, 2023 surveys),
                           definition crosswalk, source manifest, and kc_validation.py
letter/                    analysis for the paper, LaTeX sources, and outputs
  el_analysis.py           2x2 table, sensitivity/specificity/J, list-age regressions, two-year links, KC metrics
  sim_and_bootstrap.py     block bootstrap (4,999 draws) and constant-hazard simulations
  el_appendix.py           online-appendix diagnostics (by measure, survey-noise tests, robustness)
  build_el_tables.py       writes Tables 1-2;  write_appendix_body.py fills the appendix text
  check_numbers.py         57 checks: every number in the text against computed output
  wordcount.py             word counts from the compiled PDF
  manuscript/              manuscript.tex (+ el.tex, ssrn.tex, appendix.tex wrappers), references.bib, tables/
  output/                  all computed results (JSON/CSV)
docs/                      Austin survey-year definition crosswalk; SHA-256 checksums of all data files
```

## Where each result comes from

| Paper item | Produced by | File |
|---|---|---|
| Table 1 (date signal: Austin annual and two-year; Kansas City) | `el_analysis.py`, `el_appendix.py` | `letter/output/el_results.json`, `appendix.json`, `kc_signal_metrics.csv` |
| Table 2 (prior list age, four outcomes, two specifications) | `el_analysis.py` | `letter/output/list_age_tests.csv` |
| Bootstrap interval for J and for the age-gradient difference; simulation | `sim_and_bootstrap.py` | `letter/output/sim_boot.json` |
| Appendix Tables A1–A5 and appendix text | `el_appendix.py`, `write_appendix_body.py` | `letter/output/appendix.json`, `letter/manuscript/tables/appA*.tex` |
| Same-date price-change magnitudes; robustness rows; missing-data bound | `austin_pipeline/` | `phase3d_output/`, `adversarial_output/`, `revision_output/` |
| Every number in the text | `check_numbers.py` | `letter/output/number_checks.csv` |

## Data and provenance

**Greater Austin.** The Funeral Consumers Alliance of Central Texas (FCACTX, formerly the Austin Memorial and Burial Information Society) publishes an annual price survey of funeral establishments in Bastrop, Caldwell, Hays, Travis and Williamson counties. `raw_fcactx_surveys/` holds the 13 publications used. They are disclosure-safe copies: the numeric content is unchanged, and PDF author metadata and any private source-person details were removed where identified. `inputs/public_source_manifest.csv` records original and public-copy hashes.

The analysis starts from an **audited transcription** of those publications (`inputs/analytic_panel_pre_adversarial.csv`: 674 establishment-year records, 73 telephone identifiers), not from automatic re-extraction of the PDFs. `apply_source_corrections.py` applies three logged, page-referenced cell corrections (`inputs/adversarial_cell_corrections.csv`) and rebuilds lags and change flags. The audit trail is:
* `source_results/`: all 45 intervals initially flagged as same-date repricing were checked against the survey pages. Five were false flags (four row-alignment errors, one date-parsing error).
* `inputs/visual_reverification.csv`: ten randomly selected final flags (seed 20260907) and two targeted cases were rechecked against the survey pages.

The audit verifies consistency with the published survey pages, not with every underlying provider document. The panel's embalming and service-composite columns are not audited and are not used in the paper. The 2026 wave is in the panel but is not used.

**Kansas City.** `kansas_city/` contains the aggregate comparison files derived from the Funeral Consumers Alliance of Greater Kansas City's 2018, 2020 and 2023 survey publications (month-level GPL dates; records the 2020 publication marks as carried forward from 2018 are excluded). The paper's Kansas City numbers are read from `kc_market_specific_associations.csv`. To regenerate those aggregates from the establishment-level transcription, run:

```sh
python kansas_city/kc_validation.py --input kansas_city/kc_later_audited_records.csv --out kansas_city/
```

> **Not yet included:** the establishment-level Kansas City transcription (`kc_later_audited_records.csv`) and the three FCA-GKC survey PDFs. FCA-GKC has authorized their redistribution; they will be added to `kansas_city/`.

## Naming note

The scripts in `austin_pipeline/` come from an earlier, longer draft of this study, and file names containing `CEP` refer to that draft. They also compute statistics the paper does not use, such as within-brand synchronization summaries. The paper uses only the outputs listed in the table above.

## Permissions and license

See `DATA_PERMISSIONS.md`. The survey publications remain the copyright of the issuing organizations and are redistributed here with their permission. Code is released under the MIT License (`LICENSE`).

## Citation

Please cite the paper (see `CITATION.cff`) and the Funeral Consumers Alliance of Central Texas and the Funeral Consumers Alliance of Greater Kansas City as the sources of the survey data.

Contact: Taylor J. Canann, Department of Economics and Finance, College of Business, Louisiana Tech University, tcanann@latech.edu
