"""Búsqueda de hiperparámetros con Optuna (TPE) sobre el train.py de un experimento.

    python tools/hpo.py correr --base E004 --espacio hpo/espacio_base.json --nombre hpo01 --trials 40 [--arranque]
    python tools/hpo.py proponer --nombre hpo01 hpo/hpo01/propuestas.json   # trials que propone el agente
    python tools/hpo.py estado --nombre hpo01                               # resumen para leer (estado.md)

- Objetivo: media de la ganancia del ENSAMBLE en los folds de selección, con semillas del bloque de búsqueda.
- --arranque: el primer trial es la configuración del experimento base (los valores de PARAMS que están en el espacio).
- El estudio se guarda en la caché (se puede retomar); en hpo/<nombre>/ quedan el espacio, trials.tsv, estado.md y
  el log. El ganador NO se adopta solo: se copia a train.py y se corre como experimento con run.py.

Formato del espacio (JSON):
  {"learning_rate": {"tipo": "float", "min": 0.005, "max": 0.3, "log": true},
   "num_leaves": {"tipo": "int", "min": 8, "max": 2048, "log": true},
   "extra_trees": {"tipo": "cat", "valores": [false, true]}}
  Además de los parámetros de LightGBM, train_and_predict acepta "num_rounds" y "submuestreo".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import optuna

import comun
import config as C
import prepare

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _storage(nombre: str) -> str:
    (C.CACHE / "hpo").mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{C.CACHE / 'hpo' / (nombre + '.db')}"


def _sugerir(trial: optuna.Trial, espacio: dict) -> dict:
    p = {}
    for k, s in espacio.items():
        if s["tipo"] == "float":
            p[k] = trial.suggest_float(k, s["min"], s["max"], log=s.get("log", False))
        elif s["tipo"] == "int":
            p[k] = trial.suggest_int(k, s["min"], s["max"], log=s.get("log", False))
        elif s["tipo"] == "cat":
            p[k] = trial.suggest_categorical(k, s["valores"])
        else:
            raise ValueError(f"tipo desconocido en el espacio: {k}")
    return p


def _en_borde(valor, s) -> str | None:
    if s["tipo"] not in ("float", "int"):
        return None
    lo, hi, v = float(s["min"]), float(s["max"]), float(valor)
    if s.get("log") and lo > 0:
        lo, hi, v = math.log(lo), math.log(hi), math.log(v)
    pos = (v - lo) / (hi - lo) if hi > lo else 0.5
    return "mínimo" if pos < 0.05 else "máximo" if pos > 0.95 else None


def escribir_estado(nombre: str) -> str:
    carpeta = comun.RAIZ / "hpo" / nombre
    estudio = optuna.load_study(study_name=nombre, storage=_storage(nombre))
    espacio = json.loads((carpeta / "espacio.json").read_text(encoding="utf-8"))
    hechos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    filas = [{"trial": t.number, "valor": round(t.value), "origen": t.user_attrs.get("origen", "tpe"),
              "segundos": round(t.user_attrs.get("segundos", 0), 1), "params": json.dumps(t.params)} for t in hechos]
    comun.escribir_tsv(carpeta / "trials.tsv", filas, ["trial", "valor", "origen", "segundos", "params"])
    lineas = [f"# Búsqueda {nombre}", "", f"Base: {estudio.user_attrs.get('base')} | trials completos: {len(hechos)} | "
              f"semillas: {estudio.user_attrs.get('semillas')}", ""]
    if hechos:
        mejores = sorted(hechos, key=lambda t: -t.value)[:5]
        lineas += ["## Mejores 5", "", "| trial | ganancia | origen | parámetros |", "|---|---|---|---|"]
        lineas += [f"| {t.number} | {comun.millones(t.value)} | {t.user_attrs.get('origen', 'tpe')} | "
                   f"`{json.dumps(t.params)}` |" for t in mejores]
        mejor = mejores[0]
        bordes = [f"{k} en el {b} del rango ({mejor.params[k]})" for k, s in espacio.items()
                  if k in mejor.params and (b := _en_borde(mejor.params[k], s))]
        lineas += ["", "## Parámetros del mejor en el borde del rango", ""]
        lineas += [f"- {b}" for b in bordes] or ["- ninguno"]
        if len(hechos) >= 5:
            try:
                imp = optuna.importance.get_param_importances(estudio)
                lineas += ["", "## Importancia (fANOVA, orientativa con pocos trials)", ""]
                lineas += [f"- {k}: {v:.2f}" for k, v in imp.items()]
            except Exception as e:  # noqa: BLE001
                lineas += ["", f"(importancias no disponibles: {e})"]
        arranque = [t for t in hechos if t.user_attrs.get("origen") == "arranque"]
        if arranque:
            lineas += ["", f"Arranque (configuración base): {comun.millones(arranque[0].value)}"]
    texto = "\n".join(lineas) + "\n"
    (carpeta / "estado.md").write_text(texto, encoding="utf-8")
    return texto


def correr(a) -> None:
    carpeta = comun.RAIZ / "hpo" / a.nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    espacio = json.loads(Path(a.espacio).read_text(encoding="utf-8"))
    (carpeta / "espacio.json").write_text(json.dumps(espacio, indent=2) + "\n", encoding="utf-8")
    mod = comun.modulo_experimento(a.base)
    semillas = prepare.semillas("busqueda", a.semillas)
    log_f = open(carpeta / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        linea = f"{time.strftime('%H:%M:%S')} {msg}"
        print(linea, flush=True)
        log_f.write(linea + "\n")
        log_f.flush()

    df = prepare.cargar()
    datos = {m: prepare.datos_fold(df, mod, m)[:3] for m in C.MESES_TEST}
    del df
    log(f"búsqueda {a.nombre} sobre {a.base}: {a.trials} trials, semillas {semillas}, espacio {list(espacio)}")

    estudio = optuna.create_study(study_name=a.nombre, storage=_storage(a.nombre), direction="maximize",
                                  load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=semillas[0]))
    estudio.set_user_attr("base", a.base)
    estudio.set_user_attr("semillas", semillas)
    if a.arranque and not estudio.trials:
        base = {**getattr(mod, "PARAMS", {}), "num_rounds": getattr(mod, "NUM_ROUNDS", None),
                "submuestreo": getattr(mod, "SUBMUESTREO_CONTINUA", None)}
        inicial = {k: base[k] for k in espacio if base.get(k) is not None}
        estudio.enqueue_trial(inicial, user_attrs={"origen": "arranque"})

    def objetivo(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = _sugerir(trial, espacio)
        ganancias = []
        for m, (train, score, y) in datos.items():
            probs = [mod.train_and_predict(train, score, s, dict(params)) for s in semillas]
            ganancias.append(prepare.ganancia_suavizada(np.mean(probs, axis=0), y)[0])
        valor = float(np.mean(ganancias))
        trial.set_user_attr("segundos", time.time() - t0)
        trial.set_user_attr("por_fold", ganancias)
        if "origen" not in trial.user_attrs:
            trial.set_user_attr("origen", "tpe")
        log(f"trial {trial.number} ({trial.user_attrs['origen']}): {comun.millones(valor)} {json.dumps(params)}")
        return valor

    estudio.optimize(objetivo, n_trials=a.trials)
    print(escribir_estado(a.nombre))
    log_f.close()
    if not a.sin_commit:
        comun.commit(f"hpo {a.nombre}: {a.trials} trials sobre {a.base}", [carpeta])


def proponer(a) -> None:
    estudio = optuna.load_study(study_name=a.nombre, storage=_storage(a.nombre))
    propuestas = json.loads(Path(a.archivo).read_text(encoding="utf-8"))
    for p in propuestas:
        estudio.enqueue_trial(p, user_attrs={"origen": "agente"})
    print(f"{len(propuestas)} trials encolados en {a.nombre}; corren en la próxima `hpo.py correr --nombre {a.nombre}`")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correr")
    c.add_argument("--base", required=True)
    c.add_argument("--espacio", required=True)
    c.add_argument("--nombre", required=True)
    c.add_argument("--trials", type=int, default=40)
    c.add_argument("--semillas", type=int, default=5)
    c.add_argument("--arranque", action="store_true")
    c.add_argument("--sin-commit", action="store_true")
    p = sub.add_parser("proponer")
    p.add_argument("--nombre", required=True)
    p.add_argument("archivo")
    e = sub.add_parser("estado")
    e.add_argument("--nombre", required=True)
    a = ap.parse_args()
    if a.cmd == "correr":
        correr(a)
    elif a.cmd == "proponer":
        proponer(a)
    else:
        print(escribir_estado(a.nombre))
    return 0


if __name__ == "__main__":
    sys.exit(main())
