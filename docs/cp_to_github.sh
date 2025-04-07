#!/bin/bash

TARGET_DIR="../aido.wiki.github"
WIKI_DIR="../aido.wiki"
COMMIT_MSG="Update Wiki $(date '+%Y-%m-%d %H:%M:%S')"

if [ ! -d "$WIKI_DIR" ]; then
    echo "Error: Target directory '$WIKI_DIR' does not exist."
    exit 1
fi

echo "Copying Wiki from $(WIKI_DIR) to $(TARGET_DIR)"
find "$WIKI_DIR" -name "*.md" -exec cp {} "$TARGET_DIR/" \;
cd $TARGET_DIR || exit 1
git add .
git commit -m "$COMMIT_MSG"  || echo "Auto-Update: No changes to commit"
git push origin_github_