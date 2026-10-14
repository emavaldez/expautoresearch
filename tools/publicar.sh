#!/usr/bin/env bash
# Trae las ideas nuevas, regenera la página, commitea el registro y lo sube a GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/ideas.py sync || true
python tools/pagina.py
for p in docs ideas.tsv resultados.tsv estado.json LECCIONES.md features_probadas.tsv experimentos hpo sellado fase0; do
  [ -e "$p" ] && git add -- "$p"
done
git diff --cached --quiet || git commit -q -m "registro: $(date '+%Y-%m-%d %H:%M')"
git pull --rebase -q || true
git push -q
echo "publicado: https://emavaldez.github.io/expautoresearch/"
