#!/usr/bin/env bash

cd /usr/app

PDF_BUILD='pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -recorder -output-directory="build" "BA.tex"'
BIB_BUILD='bibtex build/BA.aux'

echo "Running pdflatex (1/3)"
eval "${PDF_BUILD}" > /dev/null


echo "Running bibtex (1/1)"
eval "${BIB_BUILD}" 1> /dev/null


echo "Running pdflatex (2/3)"
eval "${PDF_BUILD}" 1> /dev/null


echo "Running pdflatex (3/3)"
eval "${PDF_BUILD}" 1> /dev/null

exit 0
