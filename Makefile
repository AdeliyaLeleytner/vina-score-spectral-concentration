PYTHON ?= python3
PDFLATEX ?= pdflatex

.PHONY: all verify evidence tables figures manuscript supplement clean

all: verify evidence tables figures manuscript supplement

verify:
	$(PYTHON) analysis/verify_inputs.py

evidence:
	JCHEMINF_REQUIRE_BUNDLED=1 $(PYTHON) analysis/build_evidence.py

tables: evidence
	$(PYTHON) analysis/make_supplement_tables.py

figures: evidence
	JCHEMINF_REQUIRE_BUNDLED=1 MPLCONFIGDIR=.matplotlib-cache $(PYTHON) analysis/make_figures.py

manuscript: figures
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript.tex

supplement: tables figures
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information.tex

clean:
	$(RM) *.aux *.log *.out
