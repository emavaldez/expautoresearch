"""El loop de v4 (PROTOCOLO.md, sección v4): recorre la cola (v4/cola.tsv, por orden) y en cada paso cambia UN valor de
CAMBIOS sobre el vigente (v4/train.py), corre las 5 semillas de screening en 201907, 201908 y 201909 y aplica la regla.
Lo que pasa el screening se confirma con 10 semillas nuevas; si se confirma, el archivo nuevo pasa a ser v4/train.py y
se commitea.

    python v4/loop.py              # sigue la cola desde donde quedó (las filas sin estado)
    python v4/loop.py --solo V3    # un paso

201910 no se puntúa en ningún paso. Cada paso deja experimentos_v4/<id>/ (train.py, log.txt, eval_*.json,
resultado.json) y una fila en v4/loop.tsv; la fila de la cola queda con su estado.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arnes  # noqa: E402

RAIZ = arnes.RAIZ
VIGENTE = RAIZ / "v4" / "train.py"
COLA = RAIZ / "v4" / "cola.tsv"
REGISTRO = RAIZ / "v4" / "loop.tsv"
ESTADO = RAIZ / "v4" / "estado.json"
LOG = RAIZ / "v4" / "loop.log"
UMBRAL = 1_000_000  # meseta: + 1 M por mes en promedio de los tres meses
COLUMNAS = ["paso", "id", "cambio", "tipo", "pred_meseta", "etapa", "d_meseta", "pareado_meseta", "se_meseta", "ganadas",
            "meses_ganados", "d_lista", "d_auc", "decision", "vigente", "segundos", "fecha"]


def leer_tsv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def escribir_tsv(ruta: Path, filas: list[dict], columnas: list[str]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def estado() -> dict:
    return json.loads(ESTADO.read_text(encoding="utf-8")) if ESTADO.exists() else {}


def prender(texto: str, cambio: str) -> str:
    """Aplica 'clave=valor; clave=valor' (valores literales de Python) sobre el diccionario CAMBIOS de train.py."""
    for parte in filter(None, (p.strip() for p in cambio.split(";"))):
        clave, valor = (x.strip() for x in parte.split("=", 1))
        nuevo = repr(ast.literal_eval(valor))
        patron = re.compile(rf'^(    "{re.escape(clave)}":\s*)(.*?)(,\s*(#.*)?)$', re.M)
        texto, n = patron.subn(lambda m: m.group(1) + nuevo + m.group(3), texto, count=1)
        if n != 1:
            raise ValueError(f"no encontré '{clave}' en CAMBIOS")
    return texto


def regla(tipo: str, c: dict, etapa: str) -> bool:
    d, p = c["delta"], c["pareado"]
    if tipo == "simplifica":
        return d > -UMBRAL and p["media"] > -2 * p["se"]
    if etapa == "screening":
        return d > UMBRAL and p["media"] > 2 * p["se"] and c["meses_ganados"] >= 2
    return d > 0 and p["media"] > 2 * p["se"]


def git_commit(mensaje: str) -> str:
    subprocess.run(["git", "-C", str(RAIZ), "add", "v4/train.py"], check=True)
    subprocess.run(["git", "-C", str(RAIZ), "commit", "-q", "-m", mensaje, "--", "v4/train.py"], check=True)
    return subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def base(log) -> str:
    """V0: el vigente tal como está (E019), con las semillas de screening."""
    carpeta = arnes.RESULTADOS / "V0"
    carpeta.mkdir(parents=True, exist_ok=True)
    if not (carpeta / "train.py").exists():
        shutil.copy(VIGENTE, carpeta / "train.py")
    arnes.evaluar("V0", "screening", log=log)
    ESTADO.write_text(json.dumps({"vigente": "V0", "paso": 0}, indent=1) + "\n", encoding="utf-8")
    return "V0"


def paso(fila: dict, vigente: str, n: int, log) -> dict:
    idea, tipo = fila["id"], fila["tipo"] or "agrega"
    # reprueba, nuevas: la regla del screening con semillas que ninguna decisión vio (arnes.EXTRA)
    primera = tipo if tipo in arnes.EXTRA else "screening"
    carpeta = arnes.RESULTADOS / idea
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "train.py").write_text(prender(VIGENTE.read_text(encoding="utf-8"), fila["cambio"]), encoding="utf-8")
    t0 = time.time()
    a = arnes.evaluar(vigente, primera, log=log)
    b = arnes.evaluar(idea, primera, log=log)
    comp = {primera: arnes.todas(a, b)}
    etapa, ok = primera, regla(tipo, comp[primera]["meseta"], "screening")
    if ok:
        log(f"{idea} pasa el {primera}: confirmación con {arnes.N_CONFIRMACION} semillas nuevas")
        a = arnes.evaluar(vigente, "confirmacion", log=log)
        b = arnes.evaluar(idea, "confirmacion", log=log)
        comp["confirmacion"] = arnes.todas(a, b)
        etapa, ok = "confirmacion", regla(tipo, comp["confirmacion"]["meseta"], "confirmacion")
    decision = "queda" if ok else "se descarta"
    c = comp[etapa]
    resultado = {"id": idea, "titulo": fila["titulo"], "cambio": fila["cambio"], "tipo": tipo, "paso": n,
                 "vigente": vigente, "pred_meseta": fila["pred_meseta"], "decision": decision, "etapa": etapa,
                 "segundos": round(time.time() - t0), "fecha": f"{datetime.now():%Y-%m-%d %H:%M}",
                 "seleccion": {"vigente": {k: arnes.media(a, k) for k in arnes.METRICAS},
                               "cambio": {k: arnes.media(b, k) for k in arnes.METRICAS}},
                 "vs_vigente": comp}
    if ok:
        shutil.copy(carpeta / "train.py", VIGENTE)
        resultado["commit"] = git_commit(f"v4: {idea} queda ({fila['titulo']})\n\nMeseta {c['meseta']['delta'] / 1e6:+.1f} M "
                                         f"por mes en 201907-201909 ({etapa}).\n\n"
                                         "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    (carpeta / "resultado.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return resultado


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo")
    a = ap.parse_args()
    log_f = open(LOG, "a", encoding="utf-8")

    def log(msg: str) -> None:
        linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(linea, flush=True)
        log_f.write(linea + "\n")
        log_f.flush()

    est = estado()
    vigente = est.get("vigente") or base(log)
    n = int(estado().get("paso", 0))
    cola = sorted(leer_tsv(COLA), key=lambda f: float(f["orden"]))
    columnas_cola = list(leer_tsv(COLA)[0].keys())
    for fila in cola:
        if (a.solo and fila["id"] != a.solo) or (not a.solo and fila["estado"]):
            continue
        n += 1
        log(f"===== paso {n}: {fila['id']} ({fila['titulo']}) | {fila['cambio']} | vigente {vigente}")
        try:
            r = paso(fila, vigente, n, log)
        except Exception:
            log(traceback.format_exc())
            r = {"decision": "error", "etapa": "", "vs_vigente": {}, "segundos": 0, "fecha": f"{datetime.now():%Y-%m-%d %H:%M}"}
        c = r["vs_vigente"].get(r["etapa"], {})
        m = c.get("meseta", {})
        fila_reg = {"paso": n, "id": fila["id"], "cambio": fila["cambio"], "tipo": fila["tipo"],
                    "pred_meseta": fila["pred_meseta"], "etapa": r["etapa"], "decision": r["decision"], "vigente": vigente,
                    "segundos": r["segundos"], "fecha": r["fecha"]}
        if m:
            fila_reg.update(d_meseta=round(m["delta"]), pareado_meseta=round(m["pareado"]["media"]),
                            se_meseta=round(m["pareado"]["se"]), ganadas=f"{m['pareado']['ganadas']}/{m['pareado']['n']}",
                            meses_ganados=m["meses_ganados"], d_lista=round(c["ganancia"]["delta"]),
                            d_auc=round(c["auc"]["delta"], 5))
            log(f"{fila['id']}: {r['decision']} ({r['etapa']}) | meseta {m['delta'] / 1e6:+.1f} M por mes "
                f"(por semilla {m['pareado']['media'] / 1e6:+.1f} ± {m['pareado']['se'] / 1e6:.1f}, meses "
                f"{m['meses_ganados']}/3) | lista {c['ganancia']['delta'] / 1e6:+.1f} M | AUC {c['auc']['delta']:+.4f}")
        escribir_tsv(REGISTRO, leer_tsv(REGISTRO) + [fila_reg], COLUMNAS)
        if r["decision"] == "queda":
            vigente = fila["id"]
        ESTADO.write_text(json.dumps({"vigente": vigente, "paso": n}, indent=1) + "\n", encoding="utf-8")
        filas = leer_tsv(COLA)
        for f in filas:
            if f["id"] == fila["id"]:
                f.update(estado=r["decision"], actualizado=f"{datetime.now():%Y-%m-%d %H:%M}")
        escribir_tsv(COLA, filas, columnas_cola)
    log(f"cola terminada; vigente {vigente}")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
