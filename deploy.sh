#!/bin/bash
# open github master branch
# git pull first
# git add .
# git commit -m 'this is a message'
# git push origin main
# ssh-add ~/.ssh/id_rsa
# vim save and quit :wq
# git remote set-url origin git@github.com:chengjun/mybook.git
# open atom master branch
jupyter-book build ../mybook/
# Publish your book's HTML manually to GitHub-pages
# publish the _site folder of the master branch to the gh-pages branch
ghp-import -n -p -f _build/html
