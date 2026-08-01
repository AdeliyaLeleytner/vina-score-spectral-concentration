PYTHON ?= python3
PDFLATEX ?= pdflatex
SOURCE_DATE_EPOCH ?= 1785542400

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
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error manuscript.tex

supplement: tables figures
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary_information.tex

clean:
	$(RM) *.aux *.log *.out
