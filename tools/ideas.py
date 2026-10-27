"""Cola de cosas a probar (ideas.tsv) y su vínculo con los issues de GitHub.

Cada idea tiene un issue: ahí van los comentarios y ahí se comenta el resultado. Las ideas llegan de dos lados:
  - el plan (experimentos.md), ya aprobado;
  - las propuestas de la página: el formulario abre un issue con la etiqueta `propuesta`. Emmanuel la aprueba
    poniéndole la etiqueta `aprobada`; recién ahí entra a la cola.

Estados: propuesta → aprobada → en-curso → probada | descartada.

    python tools/ideas.py sync                        # trae propuestas y aprobaciones de GitHub a ideas.tsv
    python tools/ideas.py issues                      # crea el issue de comentarios de las ideas que no lo tienen
    python tools/ideas.py lista [--todas]             # la cola: aprobadas y en curso, por prioridad
    python tools/ideas.py estado <id> <estado> [--exp E004] [--nota "..."]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

from comun import COLUMNAS_IDEAS, IDEAS, ahora, escribir_tsv, leer_tsv
import config as C

ESTADOS = ["propuesta", "aprobada", "en-curso", "probada", "descartada"]
ORDEN_PRIORIDAD = {"base": -1, "fase 0": 0, "alta": 1, "media": 2, "baja": 3}
PAGINA = "https://expautoresearch.vercel.app"


def _gh(*args: str) -> str | None:
    if not shutil.which("gh"):
        return None
    r = subprocess.run(["gh", *args, "-R", C.REPO_GITHUB], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _campo(cuerpo: str, nombre: str) -> str:
    """Un campo del formulario de issues de GitHub (### Campo) o del formato de la página (**Campo**)."""
    m = re.search(rf"(?:\*\*|###\s*){nombre}[^\n*]*(?:\*\*)?:?\s*\n?(.+?)(?=\n(?:\*\*|###)|\Z)", cuerpo or "", re.S | re.I)
    v = m.group(1).strip() if m else ""
    return "" if v == "_No response_" else v


def sync() -> int:
    salida = _gh("issue", "list", "--label", "propuesta", "--state", "all", "--limit", "500",
                 "--json", "number,title,body,state,labels,url,createdAt,author")
    if salida is None:
        print("No pude leer los issues (¿está instalado y logueado `gh`?). La cola queda como estaba.")
        return 0
    filas = leer_tsv(IDEAS)
    por_issue = {f["issue"]: f for f in filas if f.get("issue")}
    nuevas = 0
    for it in json.loads(salida):
        n = str(it["number"])
        etiquetas = {l["name"] for l in it.get("labels", [])}
        cuerpo = it.get("body") or ""
        fila = por_issue.get(n)
        if fila is None:
            fila = {"id": f"#{n}", "origen": f"propuesta de {it.get('author', {}).get('login', '?')}", "issue": n,
                    "experimentos": "", "estado": "propuesta"}
            filas.append(fila)
            por_issue[n] = fila
            nuevas += 1
        if fila.get("estado") in ("propuesta", "aprobada", ""):
            fila["estado"] = "aprobada" if "aprobada" in etiquetas else "propuesta"
        if "descartada" in etiquetas:
            fila["estado"] = "descartada"
        prioridad = (_campo(cuerpo, "Prioridad") or "media").lower().split()[0]
        fuente = _campo(cuerpo, "Fuente")
        url = re.search(r"https?://\S+", fuente)
        fila.update({
            "titulo": re.sub(r"^\[(idea|propuesta)\]\s*", "", it["title"], flags=re.I),
            "detalle": " ".join(x for x in (_campo(cuerpo, "Qué probar"), _campo(cuerpo, "Por qué")) if x) or cuerpo.strip(),
            "fuente": fuente, "fuente_url": url.group(0).rstrip(").,") if url else "",
            "como_medir": _campo(cuerpo, "Cómo medirlo") or "Ganancia de las tres listas (202007-202009) contra el vigente.",
            "prioridad": prioridad if prioridad in ORDEN_PRIORIDAD else "media",
            "actualizado": ahora(),
        })
    escribir_tsv(IDEAS, filas, COLUMNAS_IDEAS)
    print(f"{nuevas} propuestas nuevas; {sum(f['estado'] == 'propuesta' for f in filas)} esperando aprobación; "
          f"{len(filas)} ideas en total")
    return nuevas


def issues() -> int:
    """Crea el issue de comentarios de cada idea del plan que todavía no tiene."""
    filas = leer_tsv(IDEAS)
    creadas = 0
    for f in filas:
        if f.get("issue"):
            continue
        cuerpo = (f"**Qué probar**\n{f['detalle']}\n\n**Fuente**\n{f['fuente']}"
                  + (f" ({f['fuente_url']})" if f.get("fuente_url") else "")
                  + f"\n\n**Cómo se mide**\n{f['como_medir']}\n\n**Prioridad**: {f['prioridad']}\n\n"
                  f"Comentarios sobre esta idea: qué cambiarías o qué variante probarías. "
                  f"El resultado se publica acá y en la página ({PAGINA}/#/ideas/{f['id']}).")
        url = _gh("issue", "create", "--title", f"[{f['id']}] {f['titulo']}", "--body", cuerpo, "--label", "idea")
        if url and (m := re.search(r"/issues/(\d+)", url)):
            f["issue"] = m.group(1)
            creadas += 1
            escribir_tsv(IDEAS, filas, COLUMNAS_IDEAS)
    print(f"{creadas} issues creados")
    return creadas


def lista(todas: bool = False) -> list[dict]:
    filas = leer_tsv(IDEAS)
    if not todas:
        filas = [f for f in filas if f["estado"] in ("aprobada", "en-curso")]
    filas.sort(key=lambda f: (ORDEN_PRIORIDAD.get(f["prioridad"], 9), not f["id"].startswith("#"), f["id"]))
    return filas


def marcar(id_: str, est: str, exp: str | None = None, nota: str | None = None) -> None:
    """Actualiza el estado de una idea y comenta el resultado en su issue."""
    if not id_:
        return
    filas = leer_tsv(IDEAS)
    fila = next((f for f in filas if f["id"] == id_ or (id_.startswith("#") and f.get("issue") == id_[1:])), None)
    if fila is None:
        print(f"(la idea {id_} no está en ideas.tsv; se registra igual en el experimento)")
        return
    fila["estado"] = est
    fila["actualizado"] = ahora()
    if exp:
        exps = [e for e in fila.get("experimentos", "").split(",") if e]
        if exp not in exps:
            exps.append(exp)
        fila["experimentos"] = ",".join(exps)
    escribir_tsv(IDEAS, filas, COLUMNAS_IDEAS)
    n = fila.get("issue")
    if n:
        texto = f"**{est}**" + (f" ({exp})" if exp else "") + (f": {nota}" if nota else "")
        if exp:
            texto += f"\n\nDetalle, código y log: {PAGINA}/#/experimentos/{exp}"
        _gh("issue", "comment", n, "--body", texto)
        if est == "en-curso":
            _gh("issue", "edit", n, "--add-label", "en-curso")
        elif est in ("probada", "descartada"):
            _gh("issue", "edit", n, "--remove-label", "en-curso", "--add-label", est)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    sub.add_parser("issues")
    l = sub.add_parser("lista")
    l.add_argument("--todas", action="store_true")
    e = sub.add_parser("estado")
    e.add_argument("id")
    e.add_argument("estado", choices=ESTADOS)
    e.add_argument("--exp")
    e.add_argument("--nota")
    a = ap.parse_args()
    if a.cmd == "sync":
        sync()
    elif a.cmd == "issues":
        issues()
    elif a.cmd == "lista":
        for f in lista(a.todas):
            print(f"{f['id']:>6} [{f['prioridad']:<6}] {f['estado']:<10} {f['titulo']}"
                  + (f"  ({f['experimentos']})" if f.get("experimentos") else ""))
    elif a.cmd == "estado":
        marcar(a.id, a.estado, a.exp, a.nota)


if __name__ == "__main__":
    sys.exit(main())
