# English translation and polishing record

Source: the revised Chinese fourth-edition manuscript. Scope: title, abstract, main text, 21 tables, 22 vector figures, captions, references, and supplementary information.

Applied local skills: nature-polishing (algorithmic paper, whole manuscript, Chinese-to-English, generic journal) and ieee-paper-writing. No journal-specific acceptance or formatting limits were assumed.

## Terminology ledger

| Chinese concept | English form |
|---|---|
| 幅值保留 | amplitude preservation / amplitude-preserving input |
| 毫伏输入 | millivolt input; mV in compact table labels |
| 诊断大类 | diagnostic superclass |
| 诊断子类 | diagnostic subclass |
| 交叉拟合 | cross-fitting / cross-fitted teacher |
| 完整 VOLT-KD | complete VOLT-KD |
| 本地 MSCA-Mamba | local MSCA-Mamba |
| 文献结果 | published / author-reported result |
| 宏平均平均精确率 | macro average precision (AP) |
| 完全匹配准确率 | subset accuracy |

PR-AUC and AP retain source-specific metric names in literature comparisons. Training and deployment parameter counts remain distinct.

## Editorial changes

- Reconstructed English subject--verb sentences and explicit links between observations and their implications.
- Preserved the Introduction's problem, prior-work gap, proposed approach, properties, and contributions.
- Preserved the Results evidence chain linking voltage probes, controlled input comparisons, model responses, knowledge learning, and deployment cost.
- Checked causal language and retained the observed HYP distillation trade-off, exploratory interval definitions, and differences between published and local protocols.
- Reviewed the completed English draft again for terminology, clarity, sentence length, and Results--Discussion repetition.
- Translated all table and figure labels; retained upper-right legends in the two previously revised classwise charts.

## Validation

- All 345 Chinese-bearing source lines were translated.
- All 21 table numerical-cell sequences match the Chinese source.
- Displayed equations are unchanged. The initial translation preserved the bibliography; the 8 October literature revision expands both language editions to 30 matching entries.
- All 22 figures are present and contain no Chinese labels.
- Current PDF: 33 pages; no overfull boxes, missing glyphs, or unresolved references.
- All pages were rendered and inspected; vector curves and charts were retained.

Translation preserves the main/SI allocation. The initial translation introduced no new experiment, reference, or scientific result. The subsequent literature revision adds ten verified references and expands Related Work; experiments and results are unchanged. Cross-language word counts are not treated as a compression statistic. The follow-up English polish focused on wording and typesetting rather than relocating scientific evidence.

## Literature revision — 8 October 2026

Four Related Work subsections now integrate compact modeling, foundation models, diagnostic supervision, amplitude information and lead availability. The English text was polished after expansion. See LITERATURE_UPDATE.md for the 30-entry reference audit and software attribution.

## Figure 1 replacement — 9 October 2026

Replaced Figure 1 in both languages with the user-supplied vector PDF, preserved byte for byte (424 vector paths; no raster images). Captions now describe panels (a)–(d). The English PDF has 33 pages and the Chinese PDF has 35 pages. Figure pages and adjacent pages were rendered and checked. No scientific text, table, equation or reference outside the Figure 1 block changed.

## Independent supplement — 9 October 2026

The previous combined English document is now split into a 24-page main manuscript (13 figures, 12 tables, 30 references) and a standalone 10-page supplement (9 figures, 9 tables, 5 locally numbered references). Supplementary sections and figures/tables use S prefixes. Main-to-supplement references explicitly name Supplementary Tables; reverse references identify the main manuscript. All 21 original table bodies, displayed equations and 22 figures are preserved across the two documents. Both compile independently without external auxiliary files. All supplementary pages and affected main pages were visually checked. The Chinese edition remains combined.
