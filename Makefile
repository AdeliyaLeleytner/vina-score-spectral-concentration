PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PDFLATEX ?= pdflatex
SOURCE_DATE_EPOCH ?= 1786320000
export SOURCE_DATE_EPOCH
HOTSPOT_XLS ?= data/source_cache/anastassiadis2011_moesm23.xls

PUBLIC_TESTS = \
	analysis/test_cross_panel_overlap_audit.py \
	analysis/test_public_docking44_missing_data_sensitivity.py \
	analysis/test_public_residual_null_audit.py \
	analysis/test_public_chemical_domain_controls.py \
	analysis/test_public_descriptor_domain_specificity.py \
	analysis/test_revision_map_decision_sensitivities.py \
	analysis/test_public_scaffold_holdout_recovery.py \
	analysis/test_public_panel_selector_controls.py \
	analysis/test_public_panel_selector_robustness.py \
	analysis/test_experimental_map_reliability.py \
	analysis/test_public_anastassiadis_identity_audit.py \
	analysis/test_public_anastassiadis_panel_validation.py \
	analysis/test_dense_davis_benchmark.py \
	analysis/test_dense_pkis2_benchmark.py \
	analysis/test_residual_mechanism_analysis.py \
	analysis/test_target_map_audit.py \
	analysis/test_build_public_core_evidence.py \
	analysis/test_build_public_core_macros.py \
	analysis/test_v4_submission_consistency.py \
	analysis/test_make_public_manuscript_figures.py

.PHONY: all submission-v5 submission-v4-draft public-evidence public-macros public-figures \
	verify-public-manifest public-test manuscript-v4 supplement-v4 \
	manuscript-v5 supplement-v5 figures-v5 v5-test verify-v5-numbers submission-figures \
	full-test release-check-v5 refresh-public-analyses clean

# The v5 submission build is the default. It regenerates and stages the figures
# from frozen results, checks load-bearing article and Supplement values, and then
# compiles both documents twice with no undefined references or overfull boxes.
# It does not create a tag or mint a DOI.
all: submission-v5

submission-v4-draft: public-test manuscript-v4 supplement-v4

public-evidence:
	$(PYTHON) analysis/build_public_core_evidence.py

public-macros: public-evidence
	$(PYTHON) analysis/build_public_core_macros.py

public-figures: public-evidence
	MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_public_manuscript_figures.py \
		--output figures/public_core

verify-public-manifest:
	$(PYTHON) -c 'import csv,hashlib; from pathlib import Path; root=Path("."); manifest=root/"results/public_core_source_manifest.csv"; digest=root/"results/public_core_source_manifest.sha256"; sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); expected=digest.read_text().split()[0]; assert sha(manifest)==expected,(sha(manifest),expected); rows=list(csv.DictReader(manifest.open())); bad=[r["path"] for r in rows if not (root/r["path"]).is_file() or int(r["bytes"])!=(root/r["path"]).stat().st_size or r["sha256"]!=sha(root/r["path"])]; assert not bad,bad; print(f"verified {len(rows)} public-core manifest rows")'

public-test: public-evidence public-macros public-figures verify-public-manifest
	$(PYTHON) -m pytest -q $(PUBLIC_TESTS)

manuscript-v4: public-macros public-figures
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript_v4_draft.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript_v4_draft.tex
	! grep -Eq 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined' manuscript_v4_draft.log

supplement-v4: public-macros public-figures
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information_v4_draft.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information_v4_draft.tex
	! grep -Eq 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined' supplementary_information_v4_draft.log

# ---------------------------------------------------------------------------
# v5 submission build
# ---------------------------------------------------------------------------

submission-v5: manuscript-v5 supplement-v5
	mkdir -p submission/files
	cp manuscript_v5.pdf submission/files/Manuscript.pdf
	cp supplementary_v5.pdf submission/files/Additional_file_1.pdf
	@echo "submission/files/ written: Manuscript.pdf and Additional_file_1.pdf"

figures-v5:
	MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_v5_figures.py
	MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_v5_graphical_abstract.py

# The v5 sources carry no generated macros: every value is typed into the .tex.
# Regenerating and staging the figures first also makes the graphical-abstract
# checks mandatory rather than conditional on an old local build.
verify-v5-numbers: submission-figures
	$(PYTHON) analysis/build_v5_report_manifest.py --verify
	$(PYTHON) -m pytest -q \
		analysis/test_v5_report_manifest.py \
		analysis/test_v5_manuscript_numbers.py \
		analysis/test_v5_supplement_numbers.py \
		analysis/test_reproduce_pdsp_derived_qap.py

v5-test: verify-v5-numbers

# In a Git checkout, test exactly the committed inventory so ignored historical
# PBAS/SEA experiments in a developer worktree cannot inflate or break the
# release gate.  A source archive has no .git directory, so pytest falls back to
# normal discovery over the files actually present in that archive.
full-test:
	$(PYTHON) -m pytest -q -rs $$(git ls-files 'analysis/test_*.py' 2>/dev/null)

# One-command local/CI release gate. The default ``all`` target remains the
# deliberately narrower report reconstruction from frozen result bundles.
release-check-v5: all full-test

manuscript-v5: verify-v5-numbers
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript_v5.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript_v5.tex
	! grep -Eq 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined' manuscript_v5.log

supplement-v5: verify-v5-numbers
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_v5.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_v5.tex
	! grep -Eq 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined' supplementary_v5.log

# Four main figures named in citation order. The exploratory pocket-volume figure
# is embedded only in the Supplement and is deliberately not staged as a main figure.
submission-figures: figures-v5
	mkdir -p submission/figures
	cp figures/v5/fig1_shared_axis.pdf       submission/figures/Fig1.pdf
	cp figures/v5/fig2_residual_structure.pdf submission/figures/Fig2.pdf
	cp figures/v5/fig3_external_agreement.pdf submission/figures/Fig3.pdf
	cp figures/v5/fig4_cost_and_core.pdf     submission/figures/Fig4.pdf
	cp figures/v5/fig_pocket_volume.pdf     submission/figures/FigS1.pdf
	$(RM) submission/figures/Fig5.pdf
	cp figures/v5/graphical_abstract.pdf     submission/figures/GraphicalAbstract.pdf
	$(PYTHON) -c 'from PIL import Image; import os; s=Image.open("figures/v5/graphical_abstract.png").convert("RGB"); W,H=920,300; k=min(W/s.width,H/s.height); n=(int(s.width*k),int(s.height*k)); c=Image.new("RGB",(W,H),s.getpixel((4,4))); c.paste(s.resize(n,Image.LANCZOS),((W-n[0])//2,(H-n[1])//2)); p="submission/figures/GraphicalAbstract_920x300.png"; c.save(p,"PNG",optimize=True); b=os.path.getsize(p); assert b<=150000,f"{b} bytes exceeds the 150,000-byte graphical-abstract limit"; print(f"graphical abstract {W}x{H}, {b/1000:.1f} kB")'
	@echo "submission/figures/ written: Fig1-Fig4, Supplementary FigS1, and graphical abstract"

# Expensive source-level refresh from the public datasets and fetched HotSpot source.
# The report build above consumes the frozen, checksum-registered bundles.
refresh-public-analyses:
	$(PYTHON) analysis/cross_panel_overlap_audit.py
	$(PYTHON) analysis/public_docking44_missing_data_sensitivity.py --random-controls 1000 --output-dir results/public_docking44_missing_data_sensitivity
	$(PYTHON) analysis/public_residual_null_audit.py --sample-size 12000 --repeats 500 --output-dir results/public_residual_null_audit
	$(PYTHON) analysis/public_chemical_domain_controls.py --repetitions 200 --output-dir results/public_chemical_domain_controls
	$(PYTHON) analysis/public_descriptor_domain_specificity.py --output-dir results/public_descriptor_domain_specificity
	$(PYTHON) analysis/revision_map_decision_sensitivities.py --sizes 200,500 --cluster-k 6,8,10 --repetitions 100 --random-panel-repeats 10000 --mw-null-repetitions 100 --seed 20260810 --dockstring-sample-size 15000 --dockstring-sample-seed 71 --output-dir results/revision_map_decision_sensitivities
	$(PYTHON) analysis/public_scaffold_holdout_recovery.py --output-dir results/public_scaffold_holdout_recovery
	$(PYTHON) analysis/public_panel_selector_controls.py --overwrite
	$(PYTHON) analysis/public_panel_selector_robustness.py --overwrite
	$(PYTHON) analysis/experimental_map_reliability.py --split-repeats 2000 --bootstraps 2000 --seed 20260810 --output results/experimental_map_reliability
	$(PYTHON) analysis/fetch_anastassiadis_supplement.py $(HOTSPOT_XLS)
	$(PYTHON) analysis/public_anastassiadis_identity_audit.py --quiet
	$(PYTHON) analysis/public_anastassiadis_panel_validation.py --source $(HOTSPOT_XLS)
	$(PYTHON) analysis/build_public_core_evidence.py
	$(PYTHON) analysis/build_public_core_macros.py
	MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_public_manuscript_figures.py --output figures/public_core

clean:
	$(RM) manuscript_v4_draft.aux manuscript_v4_draft.log manuscript_v4_draft.out
	$(RM) supplementary_information_v4_draft.aux supplementary_information_v4_draft.log supplementary_information_v4_draft.out
	$(RM) manuscript.aux manuscript.log manuscript.out
	$(RM) supplementary_information.aux supplementary_information.log supplementary_information.out
	$(RM) manuscript_v5.aux manuscript_v5.log manuscript_v5.out
	$(RM) supplementary_v5.aux supplementary_v5.log supplementary_v5.out supplementary_v5.toc
