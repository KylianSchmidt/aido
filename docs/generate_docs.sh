#!/bin/bash
set -e  # Exit on error

# Setup directories
WIKI_DIR="../aido.wiki"
mkdir -p "$WIKI_DIR"/{api,guides}

make html

cp home.md "$WIKI_DIR/"
cp guides/*.md "$WIKI_DIR/guides/" 2>/dev/null || true

if [ -d "_build/html/source" ]; then
    for module in _build/html/source/aido.*.html; do
        if [ -f "$module" ]; then
            basename=$(basename "$module" .html)
            python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" \
                < "$module" > "$WIKI_DIR/api/$basename.md"
        fi
    done
fi \
                < "$guide" > "$WIKI_DIR/guides/$basename.md"
        fi
    done
fi