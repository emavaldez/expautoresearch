"""El loop de autoresearch (H11, ver PROTOCOLO.md): recorre la cola de hipotesis.tsv (grupo loop, por orden) y en cada
paso prende UN interruptor de CAMBIOS sobre el vigente (v3/train.py), corre 5 semillas sólo en validación y aplica la
regla del protocolo. Si el cambio queda, el archivo nuevo pasa a ser v3/train.py y se commitea; si no, se descarta.

    python v3/loop.py               # sigue la cola desde donde quedó (v3/loop.tsv)
    python v3/loop.py --solo H2     # un paso

201910 no se puntúa en ningún paso. Cada paso deja experimentos_v3/<id>/ (train.py, log.txt, resultado.json) y una
fila en v3/loop.tsv; la hipótesis queda marcada en hipotesis.tsv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arnes  # noqa: E402
from correr import cargar_modulo  # noqa: E402

RAIZ = arnes.RAIZ
VIGENTE = RAIZ / "v3" / "train.py"
REGISTRO = RAIZ / "v3" / "loop.tsv"
HIPOTESIS = RAIZ / "hipotesis.tsv"

SIMPLIFICACIONES = {"H5", "H13", "H23", "H31"}   # sacan algo: alcanza con que no empeoren
POR_GANANCIA = {"H8", "H14", "H30"}              # pensadas para la ganancia: deciden por meseta, la AUC es el freno
VARIANTES = {"H24": ["internet0", "ctrx", "tarjeta"]}
UMBRAL_AUC, PISO_AUC, UMBRAL_MESETA = 0.001, -0.0005, 2_000_000
COLUMNAS = ["paso", "id", "variante", "pred_auc", "auc_val", "d_auc", "pareado_auc", "se_auc", "d_meseta",
            "pareado_meseta", "se_meseta", "decision", "vigente", "corrida", "segundos", "fecha"]


def leer_tsv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def escribir_tsv(ruta: Path, filas: list[dict], columnas: list[str]) -> None:
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def prender(src: str, idea: str, valor) -> str:
    """El código del vigente con el interruptor de la idea prendido (exactamente una entrada de CAMBIOS)."""
    patron = re.compile(rf'^(\s*"{re.escape(idea)}": )(False|None),', re.M)
    if len(patron.findall(src)) != 1:
        raise ValueError(f"{idea}: no encuentro su interruptor apagado en CAMBIOS")
    return patron.sub(lambda m: f"{m.group(1)}{valor!r},", src, count=1)


def decidir(idea: str, c: dict) -> str:
    a, g = c["auc"], c["ganancia_meseta"]
    pa, pg = a["pareado"], g["pareado"]
    if idea in SIMPLIFICACIONES:
        ok = a["delta"] > PISO_AUC and pa["media"] > -2 * pa["se"]
    elif idea in POR_GANANCIA:
        ok = g["delta"] > UMBRAL_MESETA and pg["media"] > 2 * pg["se"] and a["delta"] > PISO_AUC
    else:
        ok = a["delta"] > UMBRAL_AUC and pa["media"] > 2 * pa["se"]
    return "queda" if ok else "se descarta"


def marcar(idea: str, **campos) -> None:
    filas = leer_tsv(HIPOTESIS)
    columnas = list(filas[0].keys())
    for f in filas:
        if f["id"] == idea:
            f.update(campos, actualizado=f"{datetime.now():%Y-%m-%d %H:%M}")
    escribir_tsv(HIPOTESIS, filas, columnas)


def git_commit(mensaje: str) -> str:
    subprocess.run(["git", "-C", str(RAIZ), "add", "v3/train.py"], check=True)
    subprocess.run(["git", "-C", str(RAIZ), "commit", "-q", "-m", mensaje, "--", "v3/train.py"], check=True)
    return subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def paso(n: int, h: dict, vigente: str, log) -> tuple[str, str]:
    """Corre una idea (todas sus variantes) contra el vigente. Devuelve (decisión, nombre de la corrida que queda)."""
    idea = h["id"]
    src = VIGENTE.read_text(encoding="utf-8")
    mejor = None
    for valor in VARIANTES.get(idea, [True]):
        nombre = idea if valor is True else f"{idea}-{valor}"
        carpeta = arnes.RESULTADOS / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        codigo = prender(src, idea, valor)
        (carpeta / "train.py").write_text(codigo, encoding="utf-8")
        t0 = time.time()
        log(f"--- paso {n}: {nombre} (predicción ΔAUC {h['pred_auc'] or '—'}) contra {vigente}")
        try:
            mod = cargar_modulo(carpeta / "train.py")
            res = arnes.correr(mod, nombre, arnes.SEMILLAS_LOOP, None, log, con_test=False)
            c = arnes.comparar(nombre, vigente, "val", arnes.SEMILLAS_LOOP, bootstrap=False)
            d = decidir(idea, c)
        except Exception as e:  # un paso roto no frena la cola: queda registrado y se sigue
            log(traceback.format_exc())
            res, c, d = {}, None, f"error: {type(e).__name__}: {e}"[:300]
        seg = round(time.time() - t0)
        out = {"id": nombre, "hipotesis": idea, "variante": None if valor is True else valor, "paso": n,
               "vigente": vigente, "decision": d, "segundos": seg, "fecha": f"{datetime.now():%Y-%m-%d %H:%M}",
               "pred_auc": h["pred_auc"], **res, "vs_vigente": c}
        (carpeta / "resultado.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        fila = {"paso": n, "id": idea, "variante": "" if valor is True else valor, "pred_auc": h["pred_auc"],
                "decision": d, "vigente": vigente, "corrida": nombre, "segundos": seg, "fecha": out["fecha"]}
        if c:
            a, g = c["auc"], c["ganancia_meseta"]
            fila.update(auc_val=round(res["val"]["auc"], 5), d_auc=round(a["delta"], 5),
                        pareado_auc=round(a["pareado"]["media"], 5), se_auc=round(a["pareado"]["se"], 5),
                        d_meseta=round(g["delta"]), pareado_meseta=round(g["pareado"]["media"]),
                        se_meseta=round(g["pareado"]["se"]))
            log(f"RESUMEN paso {n} {nombre}: ΔAUC {a['delta']:+.4f} (pareado {a['pareado']['media']:+.4f} ± "
                f"{a['pareado']['se']:.4f}), Δmeseta {g['delta'] / 1e6:+.1f} M → {d} ({seg} s)")
        escribir_tsv(REGISTRO, leer_tsv(REGISTRO) + [fila], COLUMNAS)
        if d == "queda" and (mejor is None or c["auc"]["delta"] > mejor[2]["auc"]["delta"]):
            mejor = (valor, nombre, c, codigo)
    if mejor is None:
        ultimo = leer_tsv(REGISTRO)[-1]
        marcar(idea, estado="descartada" if not ultimo["decision"].startswith("error") else "error",
               experimento=ultimo["corrida"], conclusion=conclusion(ultimo))
        return "se descarta", vigente
    valor, nombre, c, codigo = mejor
    VIGENTE.write_text(codigo, encoding="utf-8")
    sha = git_commit(f"loop v3, paso {n}: {nombre} queda (ΔAUC val {c['auc']['delta']:+.4f})\n\n"
                     f"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    filas = leer_tsv(REGISTRO)
    marcar(idea, estado="quedó", experimento=nombre,
           conclusion=conclusion(next(f for f in filas if f["corrida"] == nombre)) + f" (commit {sha})")
    log(f"QUEDA {nombre}: nuevo vigente (commit {sha})")
    return "queda", nombre


def conclusion(f: dict) -> str:
    if not f.get("d_auc"):
        return f["decision"]
    return (f"En 201908, contra el vigente: AUC {float(f['d_auc']):+.4f} (por semilla {float(f['pareado_auc']):+.4f} ± "
            f"{float(f['se_auc']):.4f}), meseta {float(f['d_meseta']) / 1e6:+.1f} M. {f['decision'].capitalize()}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo")
    a = ap.parse_args()
    archivo_log = open(RAIZ / "v3" / "loop.log", "a", encoding="utf-8")

    def log(msg):
        linea = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(linea, flush=True)
        archivo_log.write(linea + "\n")
        archivo_log.flush()

    hechas = {f["id"] for f in leer_tsv(REGISTRO)}
    quedaron = [f for f in leer_tsv(REGISTRO) if f["decision"] == "queda"]
    vigente = quedaron[-1]["corrida"] if quedaron else "B0"
    cola = sorted((h for h in leer_tsv(HIPOTESIS) if h["fase"] == "loop" and h["estado"] == "preregistrada"
                   and h["id"] not in hechas and h["id"] != "H28"), key=lambda h: int(h["orden"]))
    if a.solo:
        cola = [h for h in cola if h["id"] == a.solo]
    n = len({f["paso"] for f in leer_tsv(REGISTRO)})
    log(f"=== loop: {len(cola)} ideas en la cola, vigente {vigente}")
    for h in cola:
        n += 1
        _, vigente = paso(n, h, vigente, log)
    log(f"=== loop terminado: vigente {vigente}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
