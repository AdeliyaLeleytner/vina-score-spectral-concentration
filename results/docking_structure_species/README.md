# Source organism of every Docking-44 receptor structure

Retrieved from the RCSB Data API (GraphQL `entries` query) for all 43 PDB
accessions in `PDB_TO_GENE`; `V1A` is a project label rather than a PDB entry.

`receptor_is_human` refers to the **receptor chain**. Five structures are not
human: `1pbq` GRIN1 (*Rattus norvegicus*), `2vt4` ADRB1 (*Meleagris gallopavo*,
8 mutations), `3kk6` PTGS1 (*Ovis aries*), `3ln1` PTGS2 (*Mus musculus*) and
`6y1z` HTR3A (*Mus musculus*).

Other non-human organisms in `all_organisms` are crystallisation aids on a human
receptor and are not species substitutions: T4 lysozyme and BRIL fusions,
llama nanobodies, murine Fab/IgG fragments, and heterotrimeric G-protein or
scFv16 chains in cryo-EM complexes. `auxiliary_chains` lists them.

Sequence-identity baselines are computed from human UniProt records throughout,
so for the five non-human receptors the docking structure and the sequence
baseline describe different species. `results/pdsp_counterscreen_retrieval`
covers 21 targets, two of which (ADRB1, HTR3A) are affected; the construct-clean
sensitivity in the manuscript drops them.
