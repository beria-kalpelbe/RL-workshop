jupyter-book clean workshop/
jupyter-book build workshop/
cd workshop/_build/html
touch .nojekyll
git init
git add .
git commit -m "Deploy book with fixed baseurl"
git branch -M gh-pages
git remote add origin https://github.com/beria-kalpelbe/RL-workshop.git
git push -f origin gh-pages
