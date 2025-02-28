#!/bin/bash

WIKI_DIR="../aido.wiki"
DOCS_DIR="../docs"
DATE_TIME=$(date "+%Y-%m-%d %H:%M:%S")

if [ -d "$WIKI_DIR" ]; then
    echo "Pulling wiki from remote"
    cd "$WIKI_DIR"
    git pull origin main || { echo "Failed to pull Wiki"; exit 1; }
else
    echo "The Wiki folder does not exist, but is necessary to push changes."
    read -n 1 -p "Do you want to clone the Wiki repository? (y/n):" choice

    if ["$choice" == "y"]; then
        echo "Cloning wiki"
        git clone git@gitlab.etp.kit.edu:kschmidt/aido.wiki.git "$WIKI_DIR"
        echo "Cloned Wiki. Re-run this script"
        exit 1
    else
        echo "Please clone the wiki 'git@gitlab.etp.kit.edu:kschmidt/aido.wiki.git' manually. Exiting"
        exit 1
    fi
fi

# Make documentation and copy to WIKI_DIR
cd "$DOCS_DIR"
bash generate_docs.sh
cp -r "$DOCS_DIR/_build/markdown/"* "$WIKI_DIR/"

# 
cd "$WIKI_DIR"
echo "Git the docs to gitlab"
git add .
git commit -m "Updated documentation $DATE_TIME" || { echo "No changes made"; exit 0; }
git push origin main || { echo "Failed to push to origin gitlab"; exit 1; }
echo "Documentation successfully uploaded";