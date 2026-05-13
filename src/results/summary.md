# Results Summary — llama8b

## Summary 1: IDs 0–253 (n=254)

| System | Accuracy | Correct / Total | Avg Qs/Case | Cases w/ 0 Qs | Total Qs | Ineffective Qs | Ineffective % |
|---|---|---|---|---|---|---|---|
| interactive | 37.80% | 96 / 254 | 0.433 | 187 / 254 (73.6%) | 110 | 54 | 49.09% |
| noninteractive_full | 64.96% | 165 / 254 | — | — | — | — | — |
| noninteractive_initial | 44.88% | 114 / 254 | — | — | — | — | — |
| scope | 51.97% | 132 / 254 | 1.697 | 1 / 254 (0.4%) | 431 | 190 | 44.08% |

> **Ineffective question**: patient responded with *"The patient cannot answer this question, please do not ask this question again."*
> Non-interactive systems ask no questions, so those columns are not applicable.
> **interactive** asked 0 questions in 73.6% of cases — the expert decided it had enough info to answer directly.

---

## Summary 2: All IDs (n=1272 for interactive/noninteractive; n=254 for scope)

| System | Accuracy | Correct / Total | Avg Qs/Case | Cases w/ 0 Qs | Total Qs | Ineffective Qs | Ineffective % |
|---|---|---|---|---|---|---|---|
| interactive | 41.75% | 531 / 1272 | 0.455 | 916 / 1272 (72.0%) | 579 | 299 | 51.64% |
| noninteractive_full | 63.84% | 812 / 1272 | — | — | — | — | — |
| noninteractive_initial | 46.46% | 591 / 1272 | — | — | — | — | — |
| scope | 51.97% | 132 / 254 | 1.697 | 1 / 254 (0.4%) | 431 | 190 | 44.08% |

> `scope` has only 254 records so both summaries are identical for it.

---

## Q/A Turn Outcome Scenarios

How each question-answer turn affects the expert's choice (per turn, not per case).

### IDs 0–253

| Scenario | interactive (110 turns) | scope (431 turns) |
|---|---|---|
| Answered → flipped to **correct** | 8 (7.3%) | 44 (10.2%) |
| Answered → flipped to **wrong** | 22 (20.0%) | 51 (11.8%) |
| Refused → flipped to **correct** | 7 (6.4%) | 55 (12.8%) |
| Refused → flipped to **wrong** | 19 (17.3%) | 49 (11.4%) |
| No change after answered | 26 (23.6%) | 146 (33.9%) |
| No change after refused | 28 (25.5%) | 86 (20.0%) |

### All IDs

| Scenario | interactive (579 turns) | scope (431 turns) |
|---|---|---|
| Answered → flipped to **correct** | 43 (7.4%) | 44 (10.2%) |
| Answered → flipped to **wrong** | 100 (17.3%) | 51 (11.8%) |
| Refused → flipped to **correct** | 42 (7.3%) | 55 (12.8%) |
| Refused → flipped to **wrong** | 116 (20.0%) | 49 (11.4%) |
| No change after answered | 137 (23.7%) | 146 (33.9%) |
| No change after refused | 141 (24.4%) | 86 (20.0%) |

> **interactive** flips to wrong far more often than to correct (~17–20% vs ~7%), while **scope** is more balanced (~12% each direction). Refused questions flip the answer roughly as often as answered ones in both systems.

---

## Runtimes — Trial Run (3 examples, timed directly)

| System | Total (3 ex.) | Avg/example | Projected: 1272 ex. | Projected: 254 ex. |
|---|---|---|---|---|
| noninteractive_full | 8.5s | ~2.3s | ~49 min | ~10 min |
| noninteractive_initial | 8.6s | ~2.4s | ~50 min | ~10 min |
| interactive | 10.6s | ~3.0s | ~64 min | ~13 min |
| scope | 117s | ~38.5s | ~816 min (~13.6 hr) | ~163 min (~2.7 hr) |

> Model load time (~1.5s) subtracted from per-example estimates.
> **scope** is ~16× slower per example than interactive, due to MCTS planning overhead in `scope_mediq_runner.py`.

---

### Notes
- **noninteractive_full**: given all context facts upfront; no questions asked.
- **noninteractive_initial**: given only initial patient info; no questions asked.
- **interactive**: expert asks questions one at a time; patient may refuse unanswerable ones.
- **scope**: similar to interactive but with scope-constrained question generation.
