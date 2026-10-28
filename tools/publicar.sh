#!/usr/bin/env bash
# Trae las ideas nuevas, regenera la página, commitea el registro y lo sube a GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" tools/ideas.py sync || true
"${PYTHON:-python3}" tools/pagina.py
for p in docs train.py ideas.tsv resultados.tsv estado.json LECCIONES.md features_probadas.tsv experimentos hpo fase0 \
         v3 experimentos_v3 hipotesis.tsv PROTOCOLO.md; do
  [ -e "$p" ] && git add -- "$p"
done
git diff --cached --quiet || git commit -q -m "registro: $(date '+%Y-%m-%d %H:%M')"
git pull --rebase --autostash -q || true
git push -q
echo "publicado: Vercel actualiza la página con este push"
