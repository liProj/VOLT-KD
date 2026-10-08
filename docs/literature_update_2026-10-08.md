# Literature update — 8 October 2026

The Chinese and English manuscripts now contain 30 references (20 previously), including 10 from 2026 and 8 from 2025. Ten papers were added: four from 2025 and six from 2026. All 30 entries are cited in the text.

## Table 3 source mapping

| Table entries | Source | Year and status |
|---|---|---|
| MSCA-Mamba | Cetintas, Diagnostics 16(19):3135; doi:10.3390/diagnostics16193135 | Published 26 September 2026 |
| 1D-ResNet18, Bidirectional Mamba, xLSTM, CWT-ViT-KAN | Mert, Akdoğan and Üvet; arXiv:2609.12803 | Preprint posted 11 September 2026 |
| MS-LTCAF | Feng et al., Bioengineering 12(9):1007; doi:10.3390/bioengineering12091007 | 2025 journal article |

These references were already present. The revised caption explicitly identifies authors, years and citation links; the Mert entry now includes its posting date. Existing reference numbers were preserved.

## Added references and their roles

| Reference | Year | Role in Related Work |
|---|---|---|
| [21] ECG-Mamba | 2025 | State-space representation and augmentation |
| [22] MAR-GCNet | 2026 | Explicit label dependency modeling |
| [23] ECG-JEPA | 2025 | Predicting latent ECG representations |
| [24] ECG-LFM | 2026 | Foundation-model objectives and downstream tasks |
| [25] Khattak et al. | 2026 | Pretraining distribution and generalization |
| [26] HMT-KD | 2026 | Hierarchical multi-teacher distillation |
| [27] SKD-AMBRNet | 2026 | Self-distillation and attention |
| [28] MDOT | 2025 | Clinical features and momentum distillation |
| [29] TolerantECG | 2025 | Noise and arbitrary lead subsets |
| [30] Su et al. | 2026 | Cross-lead distillation for binary screening |

Four Related Work subsections connect these approaches to VOLT-KD's retained voltage information, diagnostic subclass supervision, compact student, and acquisition-condition experiments. The English revision was polished for consistent terminology and evidence-bounded comparisons. New literature scores were not imported into the benchmark table without protocol verification.

## Verification

Bibliographic fields were extracted from Crossref and checked against publisher or author-hosted records. TolerantECG uses its ACM Multimedia proceedings DOI rather than the preprint. MDOT follows its recorded online publication date (19 December 2025); volume 9 and article 41 are publisher-deposited fields. Sources and claim mapping are in sources/reference_audit_2025_2026.json, and reusable BibTeX is in sources/new_references_2025_2026.bib.

Both PDFs contain 33 pages. All 21 table bodies and displayed equations match the preceding versions. All 22 figure references are unchanged. There are no undefined citations, missing glyphs, overfull boxes or detected margin violations. Revised Related Work, Table 3 and reference pages were rendered and visually checked. The bundled citation validator required an unavailable requests dependency; metadata, citation coverage, duplicate keys, year counts, table integrity and compilation were checked directly by the recorded audit instead.

## Software used for reference preparation

The installed citation-management skill guided metadata extraction and verification. Software reference: Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. 2026. https://doi.org/10.48550/arXiv.2609.00065 (record checked 8 October 2026). This software citation is documented here and is separate from the manuscript's 30 scientific references.
