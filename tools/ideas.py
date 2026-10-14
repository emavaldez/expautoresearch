"""Cola de cosas a probar (ideas.tsv).

Las ideas llegan de tres lados: el plan (experimentos.md), la página (cada propuesta es un issue de GitHub con la
etiqueta `idea`) y la línea de comandos.

    python tools/ideas.py sync                        # trae los issues con etiqueta "idea" a ideas.tsv (usa gh)
    python tools/ideas.py lista [--todas]             # pendientes, por prioridad
    python tools/ideas.py nueva "título" --detalle "qué y por qué" [--fuente ...] [--prioridad alta] [--issue]
    python tools/ideas.py estado <id> <pendiente|en-curso|probada|descartada> [--exp E004] [--nota "..."]
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

ESTADOS = ["pendiente", "en-curso", "probada", "descartada"]
ORDEN_PRIORIDAD = {"fase 0": 0, "alta": 1, "media": 2, "baja": 3}


def _gh(*args: str) -> str | None:
    if not shutil.which("gh"):
        return None
    r = subprocess.run(["gh", *args, "-R", C.REPO_GITHUB], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _campo(cuerpo: str, nombre: str) -> str:
    # acepta el formato de la página (**Campo**) y el del formulario de issues de GitHub (### Campo)
    m = re.search(rf"(?:\*\*|###\s*){nombre}[^\n*]*(?:\*\*)?:?\s*\n?(.+?)(?=\n(?:\*\*|###)|\Z)", cuerpo or "", re.S | re.I)
    return m.group(1).strip() if m else ""


def sync() -> int:
    salida = _gh("issue", "list", "--label", "idea", "--state", "all", "--limit", "500",
                 "--json", "number,title,body,state,labels,url,createdAt")
    if salida is None:
        print("No pude leer los issues (¿está instalado y logueado `gh`?). La cola queda como estaba.")
        return 0
    filas = leer_tsv(IDEAS)
    por_id = {f["id"]: f for f in filas}
    nuevas = 0
    for it in json.loads(salida):
        id_ = f"#{it['number']}"
        etiquetas = {l["name"] for l in it.get("labels", [])}
        cuerpo = it.get("body") or ""
        prioridad = (_campo(cuerpo, "Prioridad") or "media").lower().split()[0]
        if it["state"] == "CLOSED":
            est = "descartada" if "descartada" in etiquetas else "probada"
        elif "en-curso" in etiquetas:
            est = "en-curso"
        else:
            est = "pendiente"
        fila = por_id.get(id_)
        if fila is None:
            fila = {"id": id_, "origen": "página", "experimentos": ""}
            filas.append(fila)
            por_id[id_] = fila
            nuevas += 1
        fila.update({
            "titulo": re.sub(r"^\[idea\]\s*", "", it["title"], flags=re.I),
            "detalle": (_campo(cuerpo, "Qué probar") + " " + _campo(cuerpo, "Por qué")).strip() or cuerpo.strip(),
            "fuente": _campo(cuerpo, "Fuente").replace("_No response_", "").strip(" -") ,
            "prioridad": prioridad if prioridad in ORDEN_PRIORIDAD else "media",
            "estado": est if fila.get("estado") not in ("probada", "descartada") or it["state"] == "CLOSED" else fila["estado"],
            "actualizado": ahora(),
        })
    escribir_tsv(IDEAS, filas, COLUMNAS_IDEAS)
    print(f"{nuevas} ideas nuevas desde la página; {len(filas)} en total")
    return nuevas


def lista(todas: bool = False) -> list[dict]:
    filas = leer_tsv(IDEAS)
    if not todas:
        filas = [f for f in filas if f["estado"] in ("pendiente", "en-curso")]
    filas.sort(key=lambda f: (ORDEN_PRIORIDAD.get(f["prioridad"], 9), f["origen"] != "página", f["id"]))
    return filas


def nueva(titulo: str, detalle: str, fuente: str = "", prioridad: str = "media", issue: bool = False) -> str:
    filas = leer_tsv(IDEAS)
    id_ = None
    if issue:
        cuerpo = f"**Qué probar**\n{detalle}\n\n**Fuente**\n{fuente or '-'}\n\n**Prioridad**: {prioridad}\n"
        url = _gh("issue", "create", "--title", f"[idea] {titulo}", "--body", cuerpo, "--label", "idea")
        if url and (m := re.search(r"/issues/(\d+)", url)):
            id_ = f"#{m.group(1)}"
    if id_ is None:
        nums = [int(f["id"][1:]) for f in filas if re.fullmatch(r"I\d+", f["id"])]
        id_ = f"I{(max(nums) + 1) if nums else 1:03d}"
    filas.append({"id": id_, "origen": "página" if id_.startswith("#") else "local", "titulo": titulo,
                  "detalle": detalle, "fuente": fuente, "prioridad": prioridad, "estado": "pendiente",
                  "experimentos": "", "actualizado": ahora()})
    escribir_tsv(IDEAS, filas, COLUMNAS_IDEAS)
    return id_


def marcar(id_: str, est: str, exp: str | None = None, nota: str | None = None) -> None:
    """Actualiza el estado de una idea; si es un issue, lo comenta y lo cierra cuando termina."""
    if not id_:
        return
    filas = leer_tsv(IDEAS)
    fila = next((f for f in filas if f["id"] == id_), None)
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
    if id_.startswith("#"):
        n = id_[1:]
        texto = f"{est}" + (f" — {exp}" if exp else "") + (f": {nota}" if nota else "")
        _gh("issue", "comment", n, "--body", texto)
        if est == "en-curso":
            _gh("issue", "edit", n, "--add-label", "en-curso")
        elif est in ("probada", "descartada"):
            _gh("issue", "edit", n, "--remove-label", "en-curso", *(["--add-label", "descartada"] if est == "descartada" else []))
            _gh("issue", "close", n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    l = sub.add_parser("lista")
    l.add_argument("--todas", action="store_true")
    n = sub.add_parser("nueva")
    n.add_argument("titulo")
    n.add_argument("--detalle", default="")
    n.add_argument("--fuente", default="")
    n.add_argument("--prioridad", default="media", choices=list(ORDEN_PRIORIDAD))
    n.add_argument("--issue", action="store_true", help="crearla también como issue de GitHub")
    e = sub.add_parser("estado")
    e.add_argument("id")
    e.add_argument("estado", choices=ESTADOS)
    e.add_argument("--exp")
    e.add_argument("--nota")
    a = ap.parse_args()
    if a.cmd == "sync":
        sync()
    elif a.cmd == "lista":
        for f in lista(a.todas):
            print(f"{f['id']:>6} [{f['prioridad']:<6}] {f['estado']:<10} {f['titulo']}"
                  + (f"  ({f['experimentos']})" if f.get("experimentos") else ""))
    elif a.cmd == "nueva":
        print(nueva(a.titulo, a.detalle, a.fuente, a.prioridad, a.issue))
    elif a.cmd == "estado":
        marcar(a.id, a.estado, a.exp, a.nota)


if __name__ == "__main__":
    sys.exit(main())
