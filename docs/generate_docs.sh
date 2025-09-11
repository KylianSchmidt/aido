export LC_ALL=en_US.UTF-8

WIKI_DIR="../aido.wiki"
DOCS_DIR="../docs"

ln -sf ../config.json config.json
ln -s $WIKI_DIR/*.md $DOCS_DIR/
make html
sphinx-build -b markdown . _build/markdown
rm $DOCS_DIR/*md

python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/genindex.html > _build/markdown/genindex.md
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/py-modindex.html > _build/markdown/py-modindex.md
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/search.html > _build/markdown/search.md

# Copy the newly generated index files (these are always auto-generated)
cp _build/markdown/genindex.md "$WIKI_DIR/"
cp _build/markdown/py-modindex.md "$WIKI_DIR/"  
cp _build/markdown/search.md "$WIKI_DIR/"