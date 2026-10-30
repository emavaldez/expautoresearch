"""Búsqueda de la franja de v4 (PROTOCOLO.md, sección v4): el vigente (v4/train.py) con la segunda etapa con
cross-fitting (franja=1), buscando el ancho de la franja, el modelo de la segunda etapa y el peso del puesto de la base
al reordenar. El general no cambia entre trials, así
que su puntaje (por mes y semilla) y la base fuera de muestra se calculan una vez y cada trial sólo entrena la segunda
etapa: segundos por trial en lugar de una hora. Ordena por la meseta media del ensamble en 201907, 201908 y 201909 con
las semillas del bloque de búsqueda.

    python v4/hpo_franja.py B2 --trials 60

Se puede retomar. Deja experimentos_v4/<nombre>/ (busqueda.tsv, hpo.json, log.txt) y agrega el mejor a v4/cola.tsv
como el próximo paso (franja=1; franja_params=...), que pasa por la misma regla que el resto con las semillas de screening.
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

ESPACIO = {
    "franja_desde": ("float", 0.3, 1.0),
    "franja_hasta": ("float", 1.5, 5.0),
    "learning_rate": ("log", 0.01, 0.1),
    "num_leaves": ("intlog", 2, 31),
    "min_data_in_leaf": ("intlog", 50, 1000),
    "feature_fraction": ("float", 0.1, 0.8),
    "lambda_l2": ("log", 0.1, 100.0),
    "rondas": ("intlog", 30, 500),
    "peso_base": ("float", 0.0, 3.0),
    "baja1": ("cat", ["fuera", "negativa", "positiva"]),
    "init": ("cat", [True, False]),
}


def sugerir(trial: optuna.Trial) -> dict:
    p = {}
    for k, (t, *r) in ESPACIO.items():
        if t == "cat":
            p[k] = trial.suggest_categorical(k, r[0])
        elif t == "intlog":
            p[k] = trial.suggest_int(k, *r, log=True)
        else:
            p[k] = trial.suggest_float(k, *r, log=t == "log")
    return p


def actuales(mod) -> dict:
    v = {**mod.PARAMS_2, "franja_desde": mod.FRANJA_REL[0], "franja_hasta": mod.FRANJA_REL[1], "rondas": mod.RONDAS_2,
         "peso_base": mod.PESO_BASE_FRANJA, "baja1": mod.FRANJA_BAJA1, "init": mod.FRANJA_INIT}
    return {k: v[k] for k in ESPACIO}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nombre")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--semillas", type=int, default=4)
    ap.add_argument("--hilos", type=int, default=4, help="hilos de LightGBM por semilla (las semillas van en paralelo)")
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
    mod = arnes.cargar_modulo(loop.VIGENTE)
    mod.PARAMS["num_threads"] = a.hilos
    original = (dict(mod.PARAMS_2), mod.FRANJA_REL, mod.RONDAS_2, mod.PESO_BASE_FRANJA, mod.FRANJA_BAJA1, mod.FRANJA_INIT)
    # Si el vigente ya tiene la franja, el ganador se compara con ella (el trial inicial); si no, con el vigente sin franja.
    ya_tenia = mod.CAMBIOS["franja"] == 1
    hilos = ThreadPoolExecutor(max_workers=a.semillas)
    semillas = prepare.semillas("busqueda", a.semillas)
    log(f"búsqueda de la franja {a.nombre} sobre {vigente}: {a.trials} trials, semillas {semillas}")

    # El puntaje del general (con sus miembros) por mes y semilla, y la base fuera de muestra: fijos entre trials.
    df = prepare.cargar()
    datos, sin_franja = {}, []
    mod.CAMBIOS["franja"] = 0
    for m in arnes.SELECCION:
        t0 = time.time()
        train, score, y, info, intermedio = prepare.datos_fold(df, mod, m)
        ps = list(hilos.map(lambda s: prepare._llamar(mod, train, score, s, None, intermedio), semillas))
        mod.fuera_de_muestra(train)
        datos[m] = (train, score, y, ps)
        sin_franja.append(prepare.ganancia_suavizada(np.mean(ps, axis=0), y)[0])
        log(f"[{m}] general y base fuera de muestra listos ({time.time() - t0:.0f} s): meseta sin franja "
            f"{sin_franja[-1] / 1e6:.1f} M")
    mod.CAMBIOS["franja"] = 1
    del df

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    arnes.PROBS.mkdir(parents=True, exist_ok=True)
    almacen = JournalStorage(JournalFileBackend(str(arnes.PROBS / f"{a.nombre}.optuna.log")))
    estudio = optuna.create_study(study_name=a.nombre, storage=almacen, direction="maximize", load_if_exists=True,
                                  sampler=optuna.samplers.TPESampler(seed=semillas[0], multivariate=True))
    if not estudio.trials:
        estudio.enqueue_trial(actuales(mod), user_attrs={"origen": "franja=1"})

    def con_franja(train, score, s, p):
        idx = np.flatnonzero(mod.en_franja(p, len(p)))
        return mod.reordenar(p, idx, mod.puntaje_franja(1, train, score, s, p, idx))

    def objetivo(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = sugerir(trial)
        mod.PARAMS_2.clear()
        mod.PARAMS_2.update(original[0])
        mod.FRANJA_REL, mod.RONDAS_2, mod.PESO_BASE_FRANJA, mod.FRANJA_BAJA1, mod.FRANJA_INIT = original[1:]
        mod.aplicar_franja(params)
        mesetas = []
        for m, (train, score, y, ps) in datos.items():
            probs = list(hilos.map(lambda sp: con_franja(train, score, *sp), zip(semillas, ps)))
            mesetas.append(prepare.ganancia_suavizada(np.mean(probs, axis=0), y)[0])
        trial.set_user_attr("por_mes", mesetas)
        log(f"trial {trial.number} ({trial.user_attrs.get('origen', 'tpe')}): {np.mean(mesetas) / 1e6:.2f} M "
            f"[{', '.join(f'{x / 1e6:.1f}' for x in mesetas)}] ({time.time() - t0:.0f} s) {json.dumps(params)}")
        return float(np.mean(mesetas))

    hechos = len([t for t in estudio.trials if t.state.is_finished()])
    estudio.optimize(objetivo, n_trials=max(0, a.trials - hechos))

    trials = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    with open(carpeta / "busqueda.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["trial", "origen", "meseta", *ESPACIO])
        for t in sorted(trials, key=lambda t: t.number):
            w.writerow([t.number, t.user_attrs.get("origen", "tpe"), round(t.value), *[t.params.get(k, "") for k in ESPACIO]])
    mejor = max(trials, key=lambda t: t.value)
    inicial = next(t for t in trials if t.user_attrs.get("origen") == "franja=1")
    params = dict(mejor.params)
    (carpeta / "hpo.json").write_text(json.dumps({"vigente": vigente, "trials": len(estudio.trials), "semillas": semillas,
                                                  "mejor_trial": mejor.number, "valor": mejor.value,
                                                  "valor_franja1": inicial.value, "valor_sin_franja": float(np.mean(sin_franja)),
                                                  "params": params,
                                                  "espacio": {k: list(v) for k, v in ESPACIO.items()}},
                                                 indent=1) + "\n", encoding="utf-8")
    # A la cola como el próximo paso; la predicción es la mitad de lo que ganó contra el vigente (en esos mismos meses y
    # con las semillas de la búsqueda).
    filas = [f for f in loop.leer_tsv(loop.COLA) if f["id"] != a.nombre or f["estado"]]
    pendientes = [float(f["orden"]) for f in filas if not f["estado"]]
    orden = (min(pendientes) if pendientes else max(float(f["orden"]) for f in filas) + 1) - 0.5
    referencia = inicial.value if ya_tenia else np.mean(sin_franja)
    pred = (mejor.value - referencia) / 2e6
    contra = (f"su franja actual ({(inicial.value - np.mean(sin_franja)) / 1e6:+.1f} M sobre el general solo)" if ya_tenia
              else f"el vigente sin franja (franja=1 con sus parámetros: {(inicial.value - np.mean(sin_franja)) / 1e6:+.1f} M)")
    filas.append({"id": a.nombre, "orden": orden, "titulo": f"Parámetros de la franja de la búsqueda {a.nombre}",
                  "cambio": ("" if ya_tenia else "franja=1; ") + f"franja_params={params!r}", "tipo": "agrega",
                  "pred_meseta": f"{pred:+.1f}", "fuente": f"v4/hpo_franja.py, trial {mejor.number} de {len(estudio.trials)}",
                  "porque": f"la búsqueda ganó {2 * pred:+.1f} M por mes contra {contra} con sus semillas",
                  "estado": "", "actualizado": ""})
    loop.escribir_tsv(loop.COLA, filas, list(filas[0].keys()))
    log(f"mejor: trial {mejor.number}, {mejor.value / 1e6:.2f} M (franja=1 {inicial.value / 1e6:.2f} M, sin franja "
        f"{np.mean(sin_franja) / 1e6:.2f} M) {params}")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
