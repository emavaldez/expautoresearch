"""Final de v4 (PROTOCOLO.md, sección v4): el vigente contra E019 en 201910 con las semillas del bloque final, y la
ablación de lo que quedó (E019 y cada paso que quedó, en el orden en que entró). Es lo único de v4 que puntúa 201910.

    python v4/final.py --semillas 20

Deja experimentos_v4/<id>/eval_final<modo>.json de cada modelo (si ya está, no lo vuelve a correr) y
experimentos_v4/FINAL/resultado.json con cada modelo contra E019 (V0): la diferencia del ensamble, la pareada por
semilla y un intervalo bootstrap de clientes para la del ensamble (un solo mes de test: es la incertidumbre que queda).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arnes  # noqa: E402
import loop  # noqa: E402
import prepare  # noqa: E402


def cadena() -> list[str]:
    """E019 y los pasos que quedaron, en orden."""
    return ["V0"] + [f["id"] for f in loop.leer_tsv(loop.REGISTRO) if f["decision"].startswith("queda")]


def bootstrap(a: Path, b: Path, reps: int = 1000, semilla: int = 800011) -> dict:
    """Meseta del ensamble de b menos la de a sobre los mismos clientes remuestreados (con reposición)."""
    za, zb = np.load(a), np.load(b)
    assert np.array_equal(za["ids"], zb["ids"]), "los dos modelos no puntuaron los mismos clientes"
    y = za["y"]
    rng = np.random.default_rng(semilla)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        d.append(prepare.ganancia_suavizada(zb["prob"][i], y[i])[0] - prepare.ganancia_suavizada(za["prob"][i], y[i])[0])
    d = np.array(d)
    return {"media": float(d.mean()), "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
            "prob_positiva": float((d > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", type=int, default=20)
    ap.add_argument("--modelos", nargs="*", help="por defecto, E019 y los pasos que quedaron")
    a = ap.parse_args()
    modelos = a.modelos or cadena()
    semillas = prepare.semillas("final", a.semillas)
    carpeta = arnes.RESULTADOS / "FINAL"
    carpeta.mkdir(parents=True, exist_ok=True)
    log_f = open(carpeta / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(linea, flush=True)
        log_f.write(linea + "\n")
        log_f.flush()

    log(f"final: {modelos} en {arnes.TEST} con {len(semillas)} semillas del bloque final")
    df = prepare.cargar()
    res = {}
    for m in modelos:
        ruta = arnes.RESULTADOS / m / f"eval_final{arnes.MODO}.json"
        if ruta.exists():
            res[m] = json.loads(ruta.read_text(encoding="utf-8"))
            continue
        mod = arnes.cargar_modulo(arnes.RESULTADOS / m / "train.py", f"train_v4_{m}")
        if arnes.PARALELO > 1:
            mod.PARAMS["num_threads"] = arnes.HILOS
        log(f"===== {m}")
        r = prepare.evaluar(mod, arnes.TEST, semillas, df=df, log=log, guardar=arnes.PROBS / m / f"final{arnes.MODO}",
                            paralelo=arnes.PARALELO)
        ruta.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        res[m] = json.loads(ruta.read_text(encoding="utf-8"))  # meses como texto, igual que los ya guardados

    mes = arnes.TEST[0]
    salida = {"mes": mes, "semillas": semillas, "modelos": {}}
    for m in modelos:
        e = res[m][str(mes)]["ensamble"]
        fila = {k: e[k] for k in arnes.METRICAS}
        if m != "V0":
            fila["contra_E019"] = arnes.todas(res["V0"], res[m])
            fila["bootstrap_meseta"] = bootstrap(arnes.PROBS / "V0" / f"final{arnes.MODO}" / f"{mes}.npz",
                                                 arnes.PROBS / m / f"final{arnes.MODO}" / f"{mes}.npz")
        salida["modelos"][m] = fila
        c = fila.get("contra_E019", {}).get("meseta")
        extra = (f" | contra E019 {c['delta'] / 1e6:+.1f} M (por semilla {c['pareado']['media'] / 1e6:+.1f} ± "
                 f"{c['pareado']['se'] / 1e6:.1f}, bootstrap 90% {fila['bootstrap_meseta']['p05'] / 1e6:+.1f} a "
                 f"{fila['bootstrap_meseta']['p95'] / 1e6:+.1f})") if c else ""
        log(f"{m}: meseta {fila['meseta'] / 1e6:.1f} M | lista {fila['ganancia'] / 1e6:.1f} M | AUC {fila['auc']:.4f}{extra}")
    (carpeta / "resultado.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
