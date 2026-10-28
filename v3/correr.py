"""Corre un experimento v3: búsqueda de hiperparámetros (opcional), semillas, métricas y comparación con el baseline.

    python v3/correr.py B0 --busqueda                    # el baseline: 40 trials por AUC en 201908 y 10 semillas
    python v3/correr.py H9 --busqueda --objetivo meseta  # la búsqueda ordenada por ganancia meseta
    python v3/correr.py H15 --busqueda --arranque defaults
    python v3/correr.py X --params experimentos_v3/B0/hpo.json --semillas 10

Usa v3/train.py tal como está (o --train otro archivo). Deja en experimentos_v3/<id>/: train.py (copia exacta),
log.txt, busqueda.tsv, hpo.json y resultado.json. Las probabilidades van a la caché (fuera del repo).
La búsqueda se puede retomar: si se corta, al relanzar sigue desde el último trial terminado.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arnes  # noqa: E402

# Los valores por defecto de LightGBM (con bagging_fraction 1 el bagging no hace nada) y los de la primera competencia.
ARRANQUES = {
    "defaults": {"num_leaves": 31, "learning_rate": 0.1, "min_data_in_leaf": 20, "feature_fraction": 1.0,
                 "bagging_fraction": 1.0},
    "primera": {"num_leaves": 31, "learning_rate": 0.027, "min_data_in_leaf": 200, "feature_fraction": 0.5,
                "bagging_fraction": 1.0},
}


def cargar_modulo(ruta: Path):
    spec = importlib.util.spec_from_file_location("train_v3", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sugerir(trial, k, spec):
    t = spec[0]
    if t == "cat":
        return trial.suggest_categorical(k, list(spec[1]))
    a, b = spec[1], spec[2]
    return {"log": lambda: trial.suggest_float(k, a, b, log=True), "float": lambda: trial.suggest_float(k, a, b),
            "int": lambda: trial.suggest_int(k, a, b), "intlog": lambda: trial.suggest_int(k, a, b, log=True)}[t]()


def busqueda(mod, carpeta: Path, nombre: str, trials: int, objetivo: str, arranque: str | None, log,
             espacio: dict | None = None) -> dict:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    espacio = espacio or mod.ESPACIO
    tr, va, y_va, _, _ = arnes.datos(mod, log)
    semilla = arnes.semillas(1)[0]
    arnes.PROBS.mkdir(parents=True, exist_ok=True)
    almacen = JournalStorage(JournalFileBackend(str(arnes.PROBS / f"{nombre}.optuna.log")))
    estudio = optuna.create_study(study_name=nombre, storage=almacen, direction="maximize", load_if_exists=True,
                                  sampler=optuna.samplers.TPESampler(seed=semilla))
    if arranque and not estudio.trials:
        estudio.enqueue_trial({k: v for k, v in ARRANQUES[arranque].items() if k in espacio})

    def f(trial):
        params = {k: sugerir(trial, k, s) for k, s in espacio.items()}
        t0 = time.time()
        p_va, _, info = mod.train_and_predict(tr, va, y_va, None, params, semilla)
        m = arnes.metricas(p_va, y_va)
        trial.set_user_attr("auc", m["auc"])
        trial.set_user_attr("ganancia_meseta", m["ganancia_meseta"])
        trial.set_user_attr("rondas", info.get("rondas"))
        trial.set_user_attr("segundos", round(time.time() - t0, 1))
        log(f"trial {trial.number}: AUC {m['auc']:.4f} meseta {m['ganancia_meseta'] / 1e6:,.1f}M "
            f"rondas {info.get('rondas')} ({time.time() - t0:.0f} s) {params}")
        return m["auc"] if objetivo == "auc" else m["ganancia_meseta"]

    hechos = len([t for t in estudio.trials if t.state.is_finished()])
    if hechos < trials:
        estudio.optimize(f, n_trials=trials - hechos)
    filas = ["trial\tauc\tganancia_meseta\trondas\tsegundos\t" + "\t".join(espacio)]
    for t in estudio.trials:
        if t.state.name != "COMPLETE":
            continue
        u = t.user_attrs
        filas.append("\t".join(str(x) for x in [t.number, round(u["auc"], 5), round(u["ganancia_meseta"]),
                                                  u["rondas"], u["segundos"]] + [t.params[k] for k in espacio]))
    (carpeta / "busqueda.tsv").write_text("\n".join(filas) + "\n", encoding="utf-8")
    mejor = estudio.best_trial
    hpo = {"objetivo": objetivo, "arranque": arranque, "trials": len(filas) - 1, "mejor_trial": mejor.number,
           "valor": mejor.value, "params": mejor.params, "espacio": {k: list(v) for k, v in espacio.items()}}
    (carpeta / "hpo.json").write_text(json.dumps(hpo, indent=1) + "\n", encoding="utf-8")
    log(f"mejor trial {mejor.number}: {objetivo} {mejor.value:,.5f} {mejor.params}")
    del tr, va
    return mejor.params


def decidir(v: dict) -> str:
    """Regla del protocolo para una comparación en 201910 (antes de Holm)."""
    a = v["auc"]
    ok = a["delta"] > 0 and a["ic"][0] > 0 and a["pareado"]["media"] > 2 * a["pareado"]["se"]
    return "mejora" if ok else "no mejora"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--hipotesis", help="id de hipotesis.tsv (por defecto, el mismo id)")
    ap.add_argument("--train", default=str(Path(__file__).resolve().parent / "train.py"))
    ap.add_argument("--busqueda", action="store_true")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--objetivo", choices=["auc", "meseta"], default="auc")
    ap.add_argument("--arranque", choices=sorted(ARRANQUES))
    ap.add_argument("--espacio", choices=["catedra", "ampliado"], default="catedra")
    ap.add_argument("--params", help="hpo.json de otra corrida (por defecto, PARAMS de train.py)")
    ap.add_argument("--semillas", type=int, default=arnes.SEMILLAS_FINAL)
    ap.add_argument("--base", default="B0", help="contra qué corrida se compara")
    a = ap.parse_args()

    carpeta = arnes.RESULTADOS / a.id
    carpeta.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.train, carpeta / "train.py")
    mod = cargar_modulo(carpeta / "train.py")
    archivo_log = open(carpeta / "log.txt", "a", encoding="utf-8")

    def log(msg):
        linea = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(linea, flush=True)
        archivo_log.write(linea + "\n")
        archivo_log.flush()

    log(f"=== {a.id}: {' '.join(sys.argv[1:])}")
    t0 = time.time()
    params = None
    if a.busqueda:
        params = busqueda(mod, carpeta, a.id, a.trials, a.objetivo, a.arranque, log,
                          mod.ESPACIO_AMPLIADO if a.espacio == "ampliado" else mod.ESPACIO)
    elif a.params:
        params = json.loads(Path(a.params).read_text(encoding="utf-8"))["params"]
    res = arnes.correr(mod, a.id, a.semillas, params, log)
    res.update({"id": a.id, "hipotesis": a.hipotesis or a.id, "fecha": f"{datetime.now():%Y-%m-%d %H:%M}",
                "segundos": round(time.time() - t0)})
    if a.id != a.base and (arnes.PROBS / f"{a.base}.npz").exists():
        res["base"] = a.base
        res["vs_base"] = arnes.comparar(a.id, a.base, "test")
        res["vs_base_val"] = arnes.comparar(a.id, a.base, "val")
        res["decision"] = decidir(res["vs_base"])
        for k in ("auc", "ganancia_meseta"):
            res["vs_base"][k]["p_holm"] = None
    (carpeta / "resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    v, t = res["val"], res["test"]
    log(f"RESUMEN {a.id}: val AUC {v['auc']:.4f} meseta {v['ganancia_meseta'] / 1e6:,.1f}M | test AUC {t['auc']:.4f} "
        f"meseta {t['ganancia_meseta'] / 1e6:,.1f}M lista {t['ganancia_lista'] / 1e6:,.1f}M"
        + (f" | vs {a.base}: ΔAUC {res['vs_base']['auc']['delta']:+.4f} {res['decision']}" if "vs_base" in res else "")
        + f" ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
