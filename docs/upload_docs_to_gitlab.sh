#!/bin/bash

WIKI_DIR="../aido.wiki"
DOCS_DIR="../docs"
DATE_TIME=$(date "+%Y-%m-%d %H:%M:%S")

# Make documentation and copy to WIKI_DIR
echo "Generate the docs"
cd "$DOCS_DIR"
bash generate_docs.sh
cp -r "$DOCS_DIR/_build/markdown/"* "$WIKI_DIR/"

# 
cd "$WIKI_DIR"
echo "Push the docs to gitlab"
git add .
git commit -m "Updated documentation $DATE_TIME" || { echo "No changes made"; exit 0; }
git push origin main || { echo "Failed to push to origin gitlab"; exit 1; }
echo "Documentation successfully uploaded";