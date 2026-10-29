"""Búsqueda de hiperparámetros de v4 (PROTOCOLO.md, sección v4) sobre el vigente (v4/train.py): Optuna (TPE),
ordenada por la meseta media del ensamble en 201907, 201908 y 201909, con las semillas del bloque de búsqueda (no se
mezclan con las del loop). Se poda un trial que después de un mes viene por debajo de la mediana.

    python v4/hpo.py B1 --trials 60

Se puede retomar: si se corta, al relanzar sigue desde el último trial terminado. Deja experimentos_v4/<nombre>/
(busqueda.tsv, hpo.json, log.txt) y agrega el mejor a v4/cola.tsv como el próximo paso (params=...), que pasa por la
misma regla que el resto con las semillas de screening.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arnes  # noqa: E402
import loop  # noqa: E402
import prepare  # noqa: E402

# (tipo, mínimo, máximo); bagging_freq queda en 1. Los topes de hojas y rondas son para que un trial no pase de una hora.
ESPACIO = {
    "num_leaves": ("intlog", 16, 512),
    "learning_rate": ("log", 0.005, 0.1),
    "num_rounds": ("intlog", 200, 2000),
    "min_data_in_leaf": ("intlog", 20, 3000),
    "feature_fraction": ("float", 0.1, 1.0),
    "bagging_fraction": ("float", 0.3, 1.0),
    "lambda_l2": ("log", 1e-3, 100.0),
    "min_gain_to_split": ("float", 0.0, 1.0),
    "extra_trees": ("cat", [False, True]),
    "submuestreo": ("log", 0.05, 0.5),
    "peso_baja1": ("float", 0.0, 1.0),
    "peso_baja2": ("log", 1.0, 40.0),
}
E025 = {"num_leaves": 254, "learning_rate": 0.0158, "num_rounds": 667, "min_data_in_leaf": 72, "feature_fraction": 0.517,
        "lambda_l2": 0.0033, "submuestreo": 0.132}  # el mejor de la búsqueda vieja (sobre 201908-201910)


def sugerir(trial: optuna.Trial) -> dict:
    p = {}
    for k, (t, *r) in ESPACIO.items():
        if t == "cat":
            p[k] = trial.suggest_categorical(k, r[0])
        elif t == "intlog":
            p[k] = trial.suggest_int(k, r[0], r[1], log=True)
        else:
            p[k] = trial.suggest_float(k, r[0], r[1], log=t == "log")
    return {**p, "bagging_freq": 1}


def actuales(mod) -> dict:
    """Los valores del vigente para las claves del espacio (el primer trial)."""
    v = {**mod.PARAMS, "num_rounds": mod.NUM_ROUNDS, "submuestreo": mod.UNDERSAMPLING, "peso_baja1": mod.PESO_BAJA1,
         "peso_baja2": mod.PESO_BAJA2, "bagging_fraction": mod.PARAMS.get("bagging_fraction", 1.0),
         "lambda_l2": mod.PARAMS.get("lambda_l2", 1e-3), "min_gain_to_split": mod.PARAMS.get("min_gain_to_split", 0.0),
         "extra_trees": mod.PARAMS.get("extra_trees", False)}
    return {k: v[k] for k in ESPACIO}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nombre")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--semillas", type=int, default=2)
    ap.add_argument("--hilos", type=int, default=9, help="hilos de LightGBM por semilla (las semillas van en paralelo)")
    a = ap.parse_args()
    carpeta = arnes.RESULTADOS / a.nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    log_f = open(carpeta / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(linea, flush=True)
        log_f.write(linea + "\n")
        log_f.flush()

    vigente = loop.estado()["vigente"]
    mod = arnes.cargar_modulo(arnes.RESULTADOS / vigente / "train.py")
    # Dos semillas a la vez con la mitad de los hilos rinden más que una tras otra con todos. Sólo la búsqueda: el
    # loop vuelve a medir el mejor con los hilos de train.py.
    mod.PARAMS["num_threads"] = a.hilos
    original = (dict(mod.PARAMS), mod.NUM_ROUNDS, mod.UNDERSAMPLING, mod.PESO_BAJA1, mod.PESO_BAJA2)
    hilos = ThreadPoolExecutor(max_workers=a.semillas)
    semillas = prepare.semillas("busqueda", a.semillas)
    df = prepare.cargar()
    datos = {m: prepare.datos_fold(df, mod, m) for m in arnes.SELECCION}
    del df
    log(f"búsqueda {a.nombre} sobre {vigente}: {a.trials} trials, semillas {semillas}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    almacen = JournalStorage(JournalFileBackend(str(arnes.PROBS / f"{a.nombre}.optuna.log")))
    arnes.PROBS.mkdir(parents=True, exist_ok=True)
    estudio = optuna.create_study(study_name=a.nombre, storage=almacen, direction="maximize", load_if_exists=True,
                                  sampler=optuna.samplers.TPESampler(seed=semillas[0], multivariate=True),
                                  pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    if not estudio.trials:
        estudio.enqueue_trial(actuales(mod), user_attrs={"origen": "vigente"})
        estudio.enqueue_trial({**actuales(mod), **E025}, user_attrs={"origen": "E025"})

    def objetivo(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = sugerir(trial)
        mod.PARAMS.clear()
        mod.PARAMS.update(original[0])
        mod.NUM_ROUNDS, mod.UNDERSAMPLING, mod.PESO_BAJA1, mod.PESO_BAJA2 = original[1:]
        mod.aplicar(params)
        mesetas = []
        for k, (m, (train, score, y, info, intermedio)) in enumerate(datos.items()):
            probs = list(hilos.map(lambda s: prepare._llamar(mod, train, score, s, None, intermedio), semillas))
            mesetas.append(prepare.ganancia_suavizada(np.mean(probs, axis=0), y)[0])
            trial.report(float(np.mean(mesetas)), k)
            if trial.should_prune():
                log(f"trial {trial.number} podado en {m}: {np.mean(mesetas) / 1e6:.1f} M ({time.time() - t0:.0f} s)")
                raise optuna.TrialPruned()
        trial.set_user_attr("por_mes", mesetas)
        trial.set_user_attr("segundos", round(time.time() - t0))
        log(f"trial {trial.number} ({trial.user_attrs.get('origen', 'tpe')}): {np.mean(mesetas) / 1e6:.2f} M "
            f"[{', '.join(f'{x / 1e6:.1f}' for x in mesetas)}] ({time.time() - t0:.0f} s) {json.dumps(params)}")
        return float(np.mean(mesetas))

    hechos = len([t for t in estudio.trials if t.state.is_finished()])
    estudio.optimize(objetivo, n_trials=max(0, a.trials - hechos))

    trials = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    with open(carpeta / "busqueda.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["trial", "origen", "meseta", *ESPACIO, "segundos"])
        for t in sorted(estudio.trials, key=lambda t: t.number):
            if t.state.is_finished():
                w.writerow([t.number, t.user_attrs.get("origen", "tpe"),
                            "" if t.value is None else round(t.value), *[t.params.get(k, "") for k in ESPACIO],
                            t.user_attrs.get("segundos", "")])
    mejor = max(trials, key=lambda t: t.value)
    inicial = next(t for t in trials if t.user_attrs.get("origen") == "vigente")
    params = {**mejor.params, "bagging_freq": 1}
    (carpeta / "hpo.json").write_text(json.dumps({"vigente": vigente, "trials": len(estudio.trials), "semillas": semillas,
                                                  "mejor_trial": mejor.number, "valor": mejor.value,
                                                  "valor_vigente": inicial.value, "params": params,
                                                  "espacio": {k: list(v) for k, v in ESPACIO.items()}},
                                                 indent=1) + "\n", encoding="utf-8")
    # A la cola, como el próximo paso; la predicción es la mitad de lo que ganó en la búsqueda (ganó en esos mismos meses).
    # Si se retoma con más trials, la fila pendiente de antes se reemplaza.
    filas = [f for f in loop.leer_tsv(loop.COLA) if f["id"] != a.nombre or f["estado"]]
    pendientes = [float(f["orden"]) for f in filas if not f["estado"]]
    orden = (min(pendientes) if pendientes else max(float(f["orden"]) for f in filas) + 1) - 0.5
    pred = (mejor.value - inicial.value) / 2e6
    filas.append({"id": a.nombre, "orden": orden, "titulo": f"Hiperparámetros de la búsqueda {a.nombre}",
                  "cambio": f"params={params!r}", "tipo": "agrega", "pred_meseta": f"{pred:+.1f}",
                  "fuente": f"v4/hpo.py, trial {mejor.number} de {len(estudio.trials)}",
                  "porque": f"la búsqueda ganó {2 * pred:+.1f} M por mes contra los parámetros del vigente con sus semillas",
                  "estado": "", "actualizado": ""})
    loop.escribir_tsv(loop.COLA, filas, list(filas[0].keys()))
    log(f"mejor: trial {mejor.number}, {mejor.value / 1e6:.2f} M (el vigente {inicial.value / 1e6:.2f} M) {params}")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
