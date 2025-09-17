#!/bin/bash
set -e  # Exit on error
export LC_ALL=en_US.UTF-8

WIKI_DIR="../aido.wiki"
DOCS_DIR="../docs"
HTML_BUILD="$DOCS_DIR/_build/html"

mkdir -p "$WIKI_DIR"/{api,guides}
rm -rf "$DOCS_DIR/_build"

sphinx-build -b html . "$HTML_BUILD"

for page in genindex py-modindex search; do
    python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" \
        < "$HTML_BUILD/$page.html" > "$WIKI_DIR/$page.md"
done

for dir in guides api; do
    if [ -d "$HTML_BUILD/$dir" ]; then
        cp -r "$HTML_BUILD/$dir/"* "$WIKI_DIR/$dir/"
    fi
done