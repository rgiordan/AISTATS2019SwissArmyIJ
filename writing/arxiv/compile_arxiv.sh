#!/bin/bash

# Copy files and compile a version that can be uploaded to the arxiv.
# Note that, prior to running this script, you must knit the real and
# synthetic experiments with the corresponding knit_*.sh.  And before doing
# this, you must set the R variable ``single_column`` to ``TRUE`` in
# ``main.tex`` and the two ``Rnw`` files.

cp -R ../appendix_pdfs .
cp ../*tex .
cp ../*bib .
cp -R ../figure .

pdflatex main.tex
bibtex main.tex
pdflatex main.tex
pdflatex main.tex
