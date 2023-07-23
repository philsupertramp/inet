#!/usr/bin/env bash
#
#
CUR_DIR="$(pwd)"

source .venv/bin/activate

cd docs/code/source

make html

rm -rf ../../../web_docs

mv _build/html ../../../web_docs

cd ../../..

touch web_docs/.nojekyll



