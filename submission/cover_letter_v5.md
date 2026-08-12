12 August 2026

Dear Editors,

Please consider our Research article, “A shared ligand-wide score axis obscures
library-dependent target geometry in docking matrices,” for publication in the
*Journal of Cheminformatics*.

Multi-target docking matrices are often interpreted as maps of relationships
between receptors. We show that their raw correlations can instead be dominated
by a ligand-wide tendency to score favourably across many targets. Removing this
offset leaves structured target geometry, but the geometry depends on the ligand
library and its agreement with external pharmacology varies across panels. The
practical message is simple: audit the shared axis, compare the residual map with
transformation-matched nulls, and estimate it from a pilot sample drawn from the
intended chemical domain.

The study combines a fixed-pose scorer comparison, matched-null calibration,
chemical-domain controls, external pharmacology and sequence baselines, and a
sampling-cost analysis. All external analyses are post hoc. On the primary PDSP
support, the residual map is descriptively closer to experiment than the raw
map after excluding Docking-44 ligands that share a PDSP connectivity block.
Residual agreement weakens with censored cells and largely disappears on chemically shifted
complete-case docking support. We report this result as exploratory and do not
present predictor fusion as a significant or headline result.

We retain these and the other mixed results that bound the interpretation, and
provide a tested command-line tool for applying the audit to another docking
matrix. We believe this combination of methodological clarity, practical
utility and reproducible cheminformatics makes the article a good fit for the
journal.

The release includes the code, frozen redistributable inputs, aggregate external
target-pair tables, tests and figure sources needed to audit the results and rebuild the manuscript. The
compound-level PDSP export is not redistributed because we could not identify a
redistribution licence. Its public access route and checksum are recorded, and
the released aggregate table is sufficient to reconstruct the primary and
complete-case permutation tables and support sensitivities. We disclose this boundary explicitly
so that the editors can
assess it against the journal's reproducibility policy.

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
