#!/bin/bash
export LC_ALL=en_US.UTF-8

WIKI_DIR="../aido.wiki"
DOCS_DIR="../docs"
API_DIR="$WIKI_DIR/api"

echo "Building documentation..."

ln -sf ../config.json config.json

PRESERVE_FILES=("Home.md" "Getting-Started.md" "User-Guide.md" "Examples.md" "API-Reference.md")
for wiki_file in $WIKI_DIR/*.md; do
    base_file=$(basename "$wiki_file")
    should_preserve=false
    
    for preserve_file in "${PRESERVE_FILES[@]}"; do
        if [[ "$base_file" == "$preserve_file" ]]; then
            should_preserve=true
            break
        fi
    done
    
    if [[ "$should_preserve" == false && ! -L "$DOCS_DIR/$base_file" ]]; then
        ln -sf "$wiki_file" "$DOCS_DIR/"
    fi
done

echo "Generating HTML documentation..."
make html

echo "Generating Markdown documentation..."
sphinx-build -b markdown . _build/markdown

rm -f $DOCS_DIR/*.md

echo "Processing API documentation..."

rm -rf "$API_DIR"
mkdir -p "$API_DIR"

if [[ -d "_build/markdown/source" ]]; then
    echo "Copying individual module files..."
    
    for rst_file in source/aido.*.rst; do
        if [[ -f "$rst_file" ]]; then
            module_name=$(basename "$rst_file" .rst)
            md_file="_build/markdown/source/${module_name}.md"
            
            if [[ -f "$md_file" ]]; then
                cp "$md_file" "$API_DIR/"
                echo "Copied $md_file to API directory"
            fi
        fi
    done
    
    if [[ -f "_build/markdown/source/modules.md" ]]; then
        cp "_build/markdown/source/modules.md" "$API_DIR/"
        echo "Copied modules.md to API directory"
    fi
else
    echo "Warning: _build/markdown/source directory not found"
fi

echo "Generating utility pages..."
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/genindex.html > "$WIKI_DIR/genindex.md"
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/py-modindex.html > "$WIKI_DIR/py-modindex.md"  
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/search.html > "$WIKI_DIR/search.md"