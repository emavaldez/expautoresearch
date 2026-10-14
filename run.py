"""Corre UN experimento con el train.py actual y deja todo registrado.

    python run.py "descripción corta" --idea E4 --hipotesis "por qué debería mejorar"
    python run.py --restaurar          # vuelve train.py al del modelo vigente (después de un descarte)

Qué hace:
  1. Copia train.py a experimentos/<id>-<slug>/ (el código exacto que corrió) y guarda el diff contra el vigente.
  2. Evalúa en los folds de selección y de control con las 5 semillas de screening. Todo lo que imprime va a log.txt.
  3. Compara contra el vigente, pareado por semilla, y decide: a_confirmar / descartado (o base si es el primero).
  4. Escribe resultado.json, agrega una fila a resultados.tsv, actualiza la idea, regenera la página y commitea.

Un "a_confirmar" se confirma con tools/comparar.py (20 semillas frescas); sólo ahí pasa a ser el vigente.
"""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "tools"))

import comun  # noqa: E402
import config as C  # noqa: E402
import ideas  # noqa: E402
import prepare  # noqa: E402


class FueraDeTiempo(Exception):
    pass


def _alarma(*_):
    raise FueraDeTiempo()


def comparar_con_vigente(res: dict, vig: dict) -> dict:
    """Diferencias contra el vigente en los folds de selección: ensamble por fold y pareado por semilla."""
    sel = [str(m) for m in C.FOLDS_SELECCION]
    deltas = {m: res["folds"][m]["ensamble"]["ganancia"] - vig["folds"][m]["ensamble"]["ganancia"] for m in sel}
    semillas = [s for s in res["folds"][sel[0]]["semillas"] if s in vig["folds"][sel[0]]["semillas"]]
    pares = []
    for s in semillas:
        a = sum(vig["folds"][m]["semillas"][s]["ganancia"] for m in sel) / len(sel)
        b = sum(res["folds"][m]["semillas"][s]["ganancia"] for m in sel) / len(sel)
        pares.append(b - a)
    n = len(pares)
    media = sum(pares) / n if n else 0.0
    se = (sum((x - media) ** 2 for x in pares) / (n - 1) / n) ** 0.5 if n > 1 else float("nan")
    return {
        "vigente": vig["id"],
        "delta_media": res["media_seleccion"] - vig["media_seleccion"],
        "delta_por_fold": deltas,
        "folds_ganados": sum(d > 0 for d in deltas.values()),
        "pareado_media": media,
        "pareado_se": se,
    }


def restaurar() -> None:
    est = comun.estado()
    if not est.get("vigente"):
        print("No hay modelo vigente todavía.")
        return
    origen = comun.carpeta_experimento(est["vigente"]) / "train.py"
    shutil.copy(origen, RAIZ / "train.py")
    print(f"train.py <- {origen.relative_to(RAIZ)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("descripcion", nargs="?")
    ap.add_argument("--idea", default="", help="id de la idea que prueba (E4, #12, I003...)")
    ap.add_argument("--hipotesis", default="")
    ap.add_argument("--restaurar", action="store_true")
    ap.add_argument("--sin-commit", action="store_true")
    ap.add_argument("--sin-pagina", action="store_true")
    a = ap.parse_args()
    if a.restaurar:
        restaurar()
        return 0
    if not a.descripcion:
        ap.error("falta la descripción del experimento")

    est = comun.estado()
    vigente = est.get("vigente")
    id_ = comun.proximo_id()
    carpeta = comun.EXPERIMENTOS / f"{id_}-{comun.slug(a.descripcion)}"
    carpeta.mkdir(parents=True)
    shutil.copy(RAIZ / "train.py", carpeta / "train.py")
    if vigente:
        anterior = (comun.carpeta_experimento(vigente) / "train.py").read_text(encoding="utf-8").splitlines(True)
        actual = (carpeta / "train.py").read_text(encoding="utf-8").splitlines(True)
        (carpeta / "cambios.diff").write_text("".join(difflib.unified_diff(
            anterior, actual, f"{vigente}/train.py", f"{id_}/train.py")), encoding="utf-8")

    log_f = open(carpeta / "log.txt", "w", encoding="utf-8")

    def log(msg: str) -> None:
        linea = f"{time.strftime('%H:%M:%S')} {msg}"
        print(linea, flush=True)
        log_f.write(linea + "\n")
        log_f.flush()

    semillas = prepare.semillas("screening")
    meses = C.FOLDS_SELECCION + C.FOLDS_CONTROL
    log(f"{id_}: {a.descripcion}")
    log(f"idea: {a.idea or '-'} | hipótesis: {a.hipotesis or '-'} | vigente: {vigente or '(ninguno: esta es la base)'}")
    log(f"folds de selección {C.FOLDS_SELECCION}, control {C.FOLDS_CONTROL} | semillas {semillas}")
    if C.SEMILLA_PRIMIGENIA == 290497:
        log("AVISO: la semilla primigenia es la de ejemplo (config.SEMILLA_PRIMIGENIA); poné la tuya")

    t0 = time.time()
    estado_exp, folds, error = None, {}, None
    signal.signal(signal.SIGALRM, _alarma)
    signal.alarm(int(C.PRESUPUESTO_S * 1.5))
    try:
        mod = comun.cargar_modulo(carpeta / "train.py", f"modelo_{id_}")
        folds = prepare.evaluar(mod, meses, semillas, log=log, guardar=C.CACHE / "probs" / id_)
    except FueraDeTiempo:
        estado_exp, error = "fuera_de_presupuesto", f"cortado a los {C.PRESUPUESTO_S * 1.5:.0f} s"
    except Exception:
        estado_exp, error = "error", traceback.format_exc()
    finally:
        signal.alarm(0)
    segundos = round(time.time() - t0, 1)
    if error:
        log(f"ERROR: {error}")

    res = {"id": id_, "carpeta": carpeta.name, "fecha": comun.ahora(), "descripcion": a.descripcion,
           "idea": a.idea, "hipotesis": a.hipotesis, "vigente_al_correr": vigente, "semillas": semillas,
           "folds_seleccion": C.FOLDS_SELECCION, "folds_control": C.FOLDS_CONTROL, "folds": folds,
           "segundos": segundos, "error": error}
    comp = None
    if not error:
        res["media_seleccion"] = prepare.media_ensamble(folds, C.FOLDS_SELECCION)
        res["media_sueltos"] = sum(
            sum(v["ganancia"] for v in folds[str(m)]["semillas"].values()) / len(semillas) for m in C.FOLDS_SELECCION
        ) / len(C.FOLDS_SELECCION)
        res["control"] = prepare.media_ensamble(folds, C.FOLDS_CONTROL) if C.FOLDS_CONTROL else None
        res["ratio_envios"] = sum(folds[str(m)]["ensamble"]["envios"] / folds[str(m)]["clientes"]
                                  for m in C.FOLDS_SELECCION) / len(C.FOLDS_SELECCION)
        if segundos > C.PRESUPUESTO_S:
            estado_exp = "fuera_de_presupuesto"
        elif vigente is None:
            estado_exp = "base"
        else:
            comp = comparar_con_vigente(res, comun.resultado(vigente))
            res["comparacion"] = comp
            pasa = comp["delta_media"] > C.UMBRAL_SCREENING and comp["folds_ganados"] >= C.MIN_FOLDS_GANADOS
            estado_exp = "a_confirmar" if pasa else "descartado"
    res["estado"] = estado_exp
    (carpeta / "resultado.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    filas = comun.leer_tsv(comun.RESULTADOS)
    filas.append({
        "id": id_, "fecha": res["fecha"], "idea": a.idea, "descripcion": a.descripcion,
        "media_seleccion": round(res.get("media_seleccion", 0)) if not error else "",
        "delta_vs_vigente": round(comp["delta_media"]) if comp else "",
        "folds_ganados": f"{comp['folds_ganados']}/{len(C.FOLDS_SELECCION)}" if comp else "",
        "control": round(res["control"]) if res.get("control") is not None else "",
        "segundos": segundos, "estado": estado_exp, "vigente_al_correr": vigente or "",
    })
    comun.escribir_tsv(comun.RESULTADOS, filas, comun.COLUMNAS_RESULTADOS)

    if estado_exp == "base":
        est["vigente"] = id_
        comun.guardar_estado(est)
    if a.idea:
        ideas.marcar(a.idea, "en-curso" if estado_exp == "a_confirmar" else "probada", id_,
                     f"{estado_exp}, media {comun.millones(res.get('media_seleccion'))}")

    resumen = f"{id_} | {estado_exp}"
    if not error:
        resumen += f" | media selección {comun.millones(res['media_seleccion'])}"
        if comp:
            resumen += (f" | Δ {comun.millones(comp['delta_media'])} vs {vigente} | folds {comp['folds_ganados']}/"
                        f"{len(C.FOLDS_SELECCION)} | pareado {comun.millones(comp['pareado_media'])} ± "
                        f"{comun.millones(comp['pareado_se'])}")
        resumen += f" | control {comun.millones(res['control'])} | {segundos:.0f} s"
    log("RESUMEN " + resumen)
    if estado_exp == "a_confirmar":
        log(f"siguiente paso: python tools/comparar.py {vigente} {id_}")
    elif estado_exp in ("descartado", "error", "fuera_de_presupuesto") and vigente:
        log("para volver al vigente: python run.py --restaurar")
    log_f.close()

    if not a.sin_pagina:
        import pagina
        pagina.generar()
    if not a.sin_commit:
        comun.commit(f"{id_}: {a.descripcion} ({estado_exp})",
                     [carpeta, comun.RESULTADOS, comun.IDEAS, comun.ESTADO, RAIZ / "docs"])
    return 0 if not error else 1


if __name__ == "__main__":
    sys.exit(main())
