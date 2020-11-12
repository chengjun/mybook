#!/bin/bash
# open github master branch
# open atom master branch
# jupyter-book build ../mybook/
# Publish your book's HTML manually to GitHub-pages
# publish the _site folder of the master branch to the gh-pages branch
ghp-import -n -p -f _build/html
