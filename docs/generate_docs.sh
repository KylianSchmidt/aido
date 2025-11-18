#!/bin/bash
export LC_ALL=en_US.UTF-8

WIKI_DIR="../aido.wiki"
DOCS_DIR="../docs"
API_DIR="$WIKI_DIR/api"
GUIDES_DIR="$WIKI_DIR/guides"
BUILD_DIR="$DOCS_DIR/_build/markdown"

mkdir -p $API_DIR
mkdir -p $GUIDES_DIR

rm $DOCS_DIR/_build -rf

sphinx-build -b html . $DOCS_DIR/_build/html
sphinx-build -b markdown . $BUILD_DIR

python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/genindex.html > "$WIKI_DIR/genindex.md"
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/search.html > "$WIKI_DIR/search.md"

cp -r $BUILD_DIR/guides/*.md $GUIDES_DIR
cp -r $BUILD_DIR/api/*.md $API_DIR
cp $BUILD_DIR/index.md $WIKI_DIR/home.md
cp $DOCS_DIR/api/toc.md $API_DIR/toc.md 
