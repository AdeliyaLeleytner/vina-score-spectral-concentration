12 August 2026

Dear Editors,

Please consider our Methodology article, “A shared ligand-wide score axis obscures
library-dependent target geometry in docking matrices,” for publication in the
*Journal of Cheminformatics*.

Multi-target docking matrices are often interpreted as maps of relationships
between receptors. We show that their raw correlations can instead be dominated
by a ligand-wide tendency to score favourably across many targets. The centred
map retains structured geometry, but that geometry depends on the ligand library
and its agreement with external pharmacology varies across panels. The practical
message is to audit both maps: diagnose the shared axis, compare the residual map
with transformation-matched nulls, and estimate it from a pilot sample drawn from
the intended chemical domain.

The study combines a fixed-pose scorer comparison, matched-null calibration,
chemical-domain controls, external pharmacology and sequence baselines, and a
sampling-cost analysis. All external analyses are post hoc. On the primary PDSP
support, the residual map is descriptively closer to experiment than the raw
map after excluding Docking-44 ligands that share a PDSP connectivity block.
Residual agreement weakens with censored cells and largely disappears on chemically shifted
complete-case docking support. We report this result as exploratory and do not
present predictor fusion as a significant or headline result.

We retain these and the other mixed results that bound the interpretation, and
provide a tested command-line tool for applying the map-construction and
diagnostic stage of the audit to another docking
matrix. We believe this combination of methodological clarity, practical
utility and an auditable report-layer package makes the article a good fit for
the journal.

The report layer of the release candidate includes the code, frozen redistributable inputs,
aggregate external target-pair tables, tests and figure sources needed to
re-evaluate every manuscript-facing statistic from checksum-identified artifacts
and rebuild both PDFs. It is not represented as source-level regeneration of
every external-data preprocessing or docking stage. The compound-level PDSP
export is not redistributed because we could not identify a redistribution
licence. Its public no-login route and checksum are recorded, and the released
aggregate table reconstructs the primary and complete-case permutation tables
and support sensitivities. A source-by-source access matrix identifies every
verified property and every remaining licence, redistribution or clean-replay
unknown. We ask the editors to determine whether the released aggregate audit
objects are sufficient for those upstream sources under the journal's
reproducibility policy.

We confirm that this work is original, is not under consideration elsewhere,
and has been approved for submission by all authors. The authors declare the
competing interests, funding and use of generative artificial intelligence in
the manuscript. No human participants, identifiable human data, human tissue or
new animal experiments were involved.

Thank you for considering our work.

Sincerely,

Adeliya Leleytner
Corresponding author
Institute for Information Transmission Problems of the Russian Academy of
Sciences (Kharkevich Institute)
a.leleytner@gmail.com
