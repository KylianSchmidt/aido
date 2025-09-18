#!/bin/bash

WIKI_DIR="../aido.wiki"
DOCS_DIR="../docs"
DATE_TIME=$(date "+%Y-%m-%d %H:%M:%S")

cd "$DOCS_DIR"
bash generate_docs.sh

# Push to gitlab
cd "$WIKI_DIR"
git add .
git commit -m "Updated documentation $DATE_TIME"
git push origin main || { echo "Failed to push to origin gitlab"; exit 1; }
