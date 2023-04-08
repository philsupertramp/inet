#!/bin/bash
#

BASE_PATH=$1
BASE_FILENAME=$2

cd /usr/app/$BASE_PATH

mkdir -p /usr/app/build
PDF_BUILD='pdflatex -synctex=1 -interaction=nonstopmode --shell-escape --enable-write18 -file-line-error -recorder -output-directory="/usr/app/build" "/usr/app/${BASE_PATH}/${BASE_FILENAME}.tex"'
BIB_BUILD='TEXMFOUTPUT=/usr/app/build bibtex /usr/app/build/${BASE_FILENAME}.aux'

if [ ! -f "/usr/app/${BASE_PATH}/${BASE_FILENAME}.tex" ]
then
  echo "File not found."
  exit
else
  echo "File found."
fi

echo "Running pdflatex (1/3)"
eval "${PDF_BUILD}" > ../build/BA-1.log


echo "Running bibtex (1/1)"
eval "${BIB_BUILD}" > ../build/bibtex.log


echo "Running pdflatex (2/3)"
eval "${PDF_BUILD}" > ../build/BA-2.log


echo "Running pdflatex (3/3)"
eval "${PDF_BUILD}" > ../build/BA-3.log

mkdir -p /usr/app/logs
mv /usr/app/build/*.log /usr/app/logs
mv /usr/app/build/${BASE_FILENAME}.pdf /usr/app/${BASE_PATH}/${BASE_FILENAME}.pdf
/bin/rm -rf /usr/app/build/*

exit 0
