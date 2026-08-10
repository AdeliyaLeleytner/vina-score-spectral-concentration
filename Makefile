PYTHON ?= python3
PDFLATEX ?= pdflatex
SOURCE_DATE_EPOCH ?= 1786320000
PKIS1_ZIP ?= downloads/source_restricted/pkis1_supplement.zip
KIRHUB_WORKBOOK ?= downloads/source_restricted/kirhub_supplement.xlsx

.PHONY: all submission-from-frozen verify evidence validate-evidence reported \
	verify-reported tables main-figures supplement-figures graphical-abstract \
	figures manuscript supplement test fetch-pkis1 fetch-kirhub fetch-source-inputs \
	refresh-residual-nulls clean

# Deterministically rebuild the submitted PDFs from the frozen matrices and
# versioned analysis artifacts. Expensive/source-restricted upstream analyses
# are documented separately and are not silently downloaded by this target.
all: submission-from-frozen

fetch-pkis1:
	$(PYTHON) analysis/fetch_pkis1_supplement.py $(PKIS1_ZIP)

fetch-kirhub:
	$(PYTHON) analysis/fetch_kirhub_supplement.py $(KIRHUB_WORKBOOK)

fetch-source-inputs: fetch-pkis1 fetch-kirhub

# Bounded scientific-artifact refresh: put all three fixed-support residual nulls
# on one fingerprinted ligand sample, then authenticate the updated source ledger.
refresh-residual-nulls:
	$(PYTHON) analysis/refresh_residual_null_evidence.py
	$(PYTHON) analysis/update_manuscript_source_manifest.py

submission-from-frozen: verify test evidence validate-evidence reported \
	verify-reported tables figures manuscript supplement

verify:
	$(PYTHON) analysis/verify_inputs.py

evidence:
	$(PYTHON) analysis/build_manuscript_evidence.py

validate-evidence: evidence
	$(PYTHON) analysis/validate_evidence.py
	$(PYTHON) -m pytest -q analysis/test_build_manuscript_evidence.py

reported: validate-evidence
	$(PYTHON) analysis/make_reported_results.py

verify-reported: tables
	$(PYTHON) analysis/verify_reported_results.py

tables: evidence reported
	$(PYTHON) analysis/make_supplement_tables.py

main-figures: evidence
	MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_manuscript_figures.py

supplement-figures:
	MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_manuscript_supplement_figures.py

graphical-abstract: evidence
	MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_graphical_abstract.py

figures: main-figures supplement-figures graphical-abstract

manuscript: tables figures verify-reported
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript.tex
	! grep -Eq 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined' manuscript.log

supplement: tables figures
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information.tex
	! grep -Eq 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined' supplementary_information.log

test:
	$(PYTHON) -m pytest -q \
		analysis/test_build_manuscript_evidence.py \
		analysis/test_descriptor_component_geometry.py \
		analysis/test_descriptor_correlation_reduction_uncertainty.py \
		analysis/test_descriptor_rank_matched_controls.py \
		analysis/test_target_blind_descriptor_control.py \
		analysis/test_released_pair_geometry_ledger.py \
		analysis/test_dockstring_chembl_ranking_benchmark.py \
		analysis/test_increment_bootstrap_intervals.py \
		analysis/test_matched_support_geometry.py \
		analysis/test_nonvina_scorer_transport.py \
		analysis/test_dockstring_vina_term_decomposition.py \
		analysis/test_cross_panel_overlap_audit.py \
		analysis/test_davis_fixed20_censoring_sensitivity.py \
		analysis/test_dense_davis_benchmark.py \
		analysis/test_dense_pkis2_benchmark.py \
		analysis/test_experimental_panel_overlap.py \
		analysis/test_fixed20_centering_panel_sensitivity.py \
		analysis/test_fixed20_estimand_decomposition.py \
		analysis/test_fixed20_reference_sensitivity.py \
		analysis/test_equivalence_margin_sensitivity.py \
		analysis/test_strict_klifs_group_qap.py \
		analysis/test_spectral_audit.py \
		analysis/test_release_hygiene.py \
		analysis/test_ranking_centering_panel_sensitivity.py
	$(PYTHON) analysis/validate_dense_benchmarks.py

clean:
	$(RM) *.aux *.log *.out
