#!/bin/zsh
# An Overleaf-ready zip of the draft: main.tex, sections/, references.bib and the
# figures' PDFs -> ../ionosphere_paper.zip (git-ignored)
cd "$(dirname "$0")"
rm -f ../ionosphere_paper.zip
zip -q -r ../ionosphere_paper.zip main.tex references.bib sections figures/*.pdf
unzip -l ../ionosphere_paper.zip | tail -1
