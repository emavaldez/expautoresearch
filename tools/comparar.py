"""Confirmación pareada entre dos experimentos con semillas frescas.

    python tools/comparar.py E003 E007                 # 20 semillas del bloque de confirmación
    python tools/comparar.py E003 E007 --semillas 10
    python tools/comparar.py E003 E007 --solo-informe  # no toca estados

Corre el train.py guardado de cada experimento (no el de la raíz) en los folds de selección y de control.
Confirma B si:
  - el ensamble de B le gana al de A en la media de los folds de selección,
  - la diferencia pareada por semilla (media de los folds de selección) supera 2 errores estándar, y
  - en el control no pierde más de 2 errores estándar.
Si B estaba "a_confirmar" y A es el vigente, B pasa a ser el vigente.
La evaluación de cada experimento se guarda (eval_<bloque>_<n>.json) y se reutiliza en la próxima comparación.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

import comun
import config as C
import prepare


def evaluacion(id_: str, semillas: list[int], bloque: str, df, log) -> dict:
    ruta = comun.carpeta_experimento(id_) / f"eval_{bloque}_{len(semillas)}.json"
    meses = C.FOLDS_SELECCION + C.FOLDS_CONTROL
    if ruta.exists():
        guardada = json.loads(ruta.read_text(encoding="utf-8"))
        if guardada["semillas"] == semillas and guardada["meses"] == meses:
            log(f"{id_}: uso la evaluación guardada ({ruta.name})")
            return guardada["folds"]
    log(f"{id_}: evaluando con {len(semillas)} semillas del bloque {bloque}")
    folds = prepare.evaluar(comun.modulo_experimento(id_), meses, semillas, df=df, log=log)
    ruta.write_text(json.dumps({"semillas": semillas, "meses": meses, "folds": folds}, indent=1) + "\n", encoding="utf-8")
    return folds


def pareado(fa: dict, fb: dict, meses: list[int], semillas: list[int]) -> dict:
    d = np.array([np.mean([fb[str(m)]["semillas"][str(s)]["ganancia"] - fa[str(m)]["semillas"][str(s)]["ganancia"]
                           for m in meses]) for s in semillas])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return {"media": float(d.mean()), "se": se, "gana": int((d > 0).sum()), "n": int(len(d)), "p_wilcoxon": p}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="referencia (normalmente el vigente)")
    ap.add_argument("b", help="candidato")
    ap.add_argument("--semillas", type=int, default=20)
    ap.add_argument("--bloque", default="confirmacion", choices=list(C.BLOQUES))
    ap.add_argument("--solo-informe", action="store_true")
    ap.add_argument("--sin-commit", action="store_true")
    a = ap.parse_args()

    carpeta_b = comun.carpeta_experimento(a.b)
    log_f = open(carpeta_b / f"confirmacion_vs_{a.a}.log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        linea = f"{time.strftime('%H:%M:%S')} {msg}"
        print(linea, flush=True)
        log_f.write(linea + "\n")
        log_f.flush()

    semillas = prepare.semillas(a.bloque, a.semillas)
    df = prepare.cargar()
    fa = evaluacion(a.a, semillas, a.bloque, df, log)
    fb = evaluacion(a.b, semillas, a.bloque, df, log)

    sel = C.FOLDS_SELECCION
    ens_a = prepare.media_ensamble(fa, sel)
    ens_b = prepare.media_ensamble(fb, sel)
    par = pareado(fa, fb, sel, semillas)
    ctrl = pareado(fa, fb, C.FOLDS_CONTROL, semillas) if C.FOLDS_CONTROL else None
    por_fold = {str(m): {"a": fa[str(m)]["ensamble"]["ganancia"], "b": fb[str(m)]["ensamble"]["ganancia"]}
                for m in sel + C.FOLDS_CONTROL}

    confirma = (ens_b > ens_a and par["media"] > 2 * par["se"]
                and (ctrl is None or ctrl["media"] > -2 * ctrl["se"]))
    decision = "confirmado" if confirma else "rechazado"
    for m, v in por_fold.items():
        log(f"[{m}] ensamble {a.a} {comun.millones(v['a'])} | {a.b} {comun.millones(v['b'])} | "
            f"Δ {comun.millones(v['b'] - v['a'])}")
    log(f"selección: ensamble Δ {comun.millones(ens_b - ens_a)} | pareado {comun.millones(par['media'])} ± "
        f"{comun.millones(par['se'])} ({par['gana']}/{par['n']} semillas, Wilcoxon p={par['p_wilcoxon']:.3f})")
    if ctrl:
        log(f"control: pareado {comun.millones(ctrl['media'])} ± {comun.millones(ctrl['se'])}")
    log(f"DECISIÓN: {decision}")

    informe = {"a": a.a, "b": a.b, "fecha": comun.ahora(), "bloque": a.bloque, "semillas": semillas,
               "ensamble_a": ens_a, "ensamble_b": ens_b, "pareado": par, "control": ctrl, "por_fold": por_fold,
               "decision": decision}
    (carpeta_b / f"confirmacion_vs_{a.a}.json").write_text(json.dumps(informe, indent=2) + "\n", encoding="utf-8")
    log_f.close()

    rutas = [carpeta_b, comun.carpeta_experimento(a.a)]
    if not a.solo_informe:
        res_b = comun.resultado(a.b)
        est = comun.estado()
        if res_b.get("estado") == "a_confirmar" and est.get("vigente") == a.a:
            res_b["estado"] = decision
            res_b["confirmacion"] = informe
            (carpeta_b / "resultado.json").write_text(json.dumps(res_b, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
            comun.actualizar_resultado_tsv(a.b, estado=decision)
            if confirma:
                est["vigente"] = a.b
                comun.guardar_estado(est)
                print(f"{a.b} es el nuevo vigente")
            else:
                print(f"{a.b} rechazado; para volver: python run.py --restaurar")
            if res_b.get("idea"):
                import ideas
                ideas.marcar(res_b["idea"], "probada", a.b, f"{decision} con {len(semillas)} semillas frescas")
            rutas += [comun.RESULTADOS, comun.ESTADO, comun.IDEAS]
        import pagina
        pagina.generar()
        rutas.append(comun.RAIZ / "docs")
    if not a.sin_commit:
        comun.commit(f"{a.b} vs {a.a}: {decision} ({len(semillas)} semillas)", rutas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
