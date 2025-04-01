export LC_ALL=en_US.UTF-8
ln -s ../config.json config.json
make html
sphinx-build -b markdown . _build/markdown

python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/genindex.html > _build/markdown/genindex.md
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/py-modindex.html > _build/markdown/py-modindex.md
python -c "import html2text, sys; print(html2text.html2text(sys.stdin.read()))" < _build/html/search.html > _build/markdown/search.md

