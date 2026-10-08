# VOLT-KD — English main manuscript

Main file: manuscript_en.tex (24 pages, 13 figures, 12 tables, 30 references).

Compile independently with XeLaTeX twice:

    xelatex -interaction=nonstopmode -halt-on-error manuscript_en.tex
    xelatex -interaction=nonstopmode -halt-on-error manuscript_en.tex

The Supplementary Methods and Complete Results section is now a separate document in ../supplement_en/supplementary_en.tex (10 pages, figures S1–S9, tables S1–S9). The main PDF no longer contains the appendix. Supplementary references in the main text retain their S numbers. Each document contains its own bibliography and required figure files; neither requires the other's auxiliary files.

VOLT-KD_English_LaTeX.zip contains the main manuscript and its 13 required figures. The independent supplementary package is VOLT-KD_Supplementary_LaTeX.zip in the supplementary directory.

The English manuscript was translated and polished using the installed nature-polishing skill. See POLISHING_NOTES.md and LITERATURE_UPDATE.md for the editorial history. Figure 1 uses the user-supplied vector main figure.

Public repository: https://github.com/liProj/VOLT-KD
