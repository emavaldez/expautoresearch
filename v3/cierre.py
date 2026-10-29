"""Cierre de los experimentos v3 (PROTOCOLO.md), sobre las corridas ya guardadas:

- H25: el final con 40 trials nuevos (H11) contra el final con los parámetros heredados del loop (FINAL0), en 201910;
- H18: el ensamble de 50 semillas contra el de las 10 primeras, en 201910;
- H36, H37 y H38: otras reglas de corte sobre el modelo final (sólo cambian la ganancia de la lista);
- H33: si las predicciones escritas antes de cada paso ordenan lo que pasó (Spearman);
- H34: cuánto se ahorraría cortando un paso cuando su primera semilla viene 0,002 abajo de la del vigente;
- Holm sobre las confirmatorias (H11, H25, H9, H10, H15, H35, H18) y la decisión de cada una.

    python v3/cierre.py

Escribe experimentos_v3/<id>/resultado.json de las que se calculan acá y completa p_holm y la decisión en las demás.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arnes  # noqa: E402
import prepare  # noqa: E402

R = arnes.RESULTADOS
CONFIRMATORIAS = {"H11": "auc", "H25": "auc", "H9": "ganancia_meseta", "H10": "auc", "H15": "auc", "H35": "auc",
                  "H18": "auc"}
UMBRAL_CALIBRADO = arnes.prepare.C.COSTO_ESTIMULO / (arnes.prepare.C.GAN_ACIERTO + arnes.prepare.C.COSTO_ESTIMULO)


def leer(nombre: str) -> dict:
    return json.loads((R / nombre / "resultado.json").read_text(encoding="utf-8"))


def escribir(nombre: str, d: dict) -> None:
    (R / nombre).mkdir(parents=True, exist_ok=True)
    (R / nombre / "resultado.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def ganancia(p, y, n) -> float:
    return arnes.metricas(p, y)["ganancia_maxima"] if n is None else float(arnes.curva(p, y)[max(1, int(n)) - 1])


def h25() -> None:
    d = {"id": "H25", "hipotesis": "H25", "base": "FINAL0", "vs_base": arnes.comparar("H11", "FINAL0", "test"),
         "vs_base_val": arnes.comparar("H11", "FINAL0", "val", bootstrap=False)}
    d["val"], d["test"] = leer("H11")["val"], leer("H11")["test"]
    escribir("H25", d)


def h18() -> None:
    """Ensamble de 50 contra el de las 10 primeras semillas de la misma corrida (son las mismas 10 de H11)."""
    A = arnes.probs("H18")
    out = {"id": "H18", "hipotesis": "H18", "base": "las 10 primeras semillas"}
    for mes in ("val", "test"):
        y = A[f"y_{mes}"]
        p50, p10 = A[mes].mean(axis=0), A[mes][:10].mean(axis=0)
        m50, m10 = arnes.metricas(p50, y), arnes.metricas(p10, y)
        out[mes] = m50
        comp = {k: {"delta": m50[k] - m10[k]} for k in ("auc", "ganancia_meseta", "ganancia_lista", "auc_parcial", "top1")}
        if mes == "test":
            b = arnes._remuestreo(p50, p10, y)
            for k in ("auc", "ganancia_meseta"):
                comp[k].update(ic=b[k]["ic"], p=b[k]["p"])
        out["vs_base" if mes == "test" else "vs_base_val"] = comp
    escribir("H18", out)


# ---------------------------------------------------------------- corte

def _segmento(mes: int) -> np.ndarray:
    """internet == 0 de los clientes del mes, en el orden en que el arnés arma validación y test."""
    df = prepare.cargar()
    df = df[(df[arnes.MES] >= arnes.PRIMER_MES) & (df[arnes.MES] <= arnes.TEST)].reset_index(drop=True)
    return (df.loc[df[arnes.MES] == mes, "internet"] == 0).to_numpy()


def _tasa(mes: int) -> float:
    df = prepare.cargar()
    c = df.loc[df[arnes.MES] == mes, arnes.CLASE].astype(str)
    return float((c == "BAJA+2").mean())


def corte(final: str = "H11") -> None:
    from sklearn.isotonic import IsotonicRegression
    A = arnes.probs(final)
    pv, yv = A["val"].mean(axis=0), A["y_val"]
    pt, yt = A["test"].mean(axis=0), A["y_test"]
    lista = {m: ganancia(p, y, round(arnes.FRACCION_LISTA * len(p))) for m, p, y in (("val", pv, yv), ("test", pt, yt))}

    def fila(nombre, hip, n_val, n_test, extra):
        g = {"val": ganancia(pv, yv, n_val), "test": ganancia(pt, yt, n_test)}
        escribir(nombre, {"id": nombre, "hipotesis": hip, "base": f"{final}, lista al 3,3%",
                          "corte": {"envios_val": int(n_val), "envios_test": int(n_test), **extra},
                          "val": {"ganancia_lista": g["val"], "envios_lista": int(n_val)},
                          "test": {"ganancia_lista": g["test"], "envios_lista": int(n_test)},
                          "vs_lista": {m: g[m] - lista[m] for m in ("val", "test")},
                          "decision": "mejora" if g["val"] > lista["val"] and g["test"] > lista["test"] else "no mejora"})

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yv.astype(float))  # se calibra en 201908
    fila("H36", "H36", (iso.predict(pv) >= UMBRAL_CALIBRADO).sum(), (iso.predict(pt) >= UMBRAL_CALIBRADO).sum(),
         {"umbral": UMBRAL_CALIBRADO})
    media_tr = float(np.mean([_tasa(m) for m in (201903, 201904, 201905, 201906)]))
    f_val = arnes.FRACCION_LISTA * _tasa(201906) / media_tr  # para 201908, el último mes con clase completa es 201906
    f_test = arnes.FRACCION_LISTA * _tasa(201908) / media_tr  # para 201910, 201908
    fila("H37", "H37", round(f_val * len(pv)), round(f_test * len(pt)), {"fraccion_val": f_val, "fraccion_test": f_test})
    sv, st = _segmento(arnes.VALIDACION), _segmento(arnes.TEST)
    grilla = np.arange(0.005, 0.1001, 0.0025)

    def por_segmento(p, y, s, f0, f1):
        idx = []
        for k, f in ((s, f0), (~s, f1)):
            i = np.flatnonzero(k)
            idx.append(i[np.argsort(-p[i], kind="stable")[:round(f * len(i))]])
        i = np.concatenate(idx)
        return float(np.where(y[i], prepare.C.GAN_ACIERTO, -prepare.C.COSTO_ESTIMULO).sum()), len(i)

    mejor = max(((por_segmento(pv, yv, sv, a, b)[0], a, b) for a in grilla for b in grilla))
    _, f0, f1 = mejor
    gv, nv = por_segmento(pv, yv, sv, f0, f1)
    gt, nt = por_segmento(pt, yt, st, f0, f1)
    escribir("H38", {"id": "H38", "hipotesis": "H38", "base": f"{final}, lista al 3,3%",
                     "corte": {"fraccion_internet0": f0, "fraccion_resto": f1, "envios_val": nv, "envios_test": nt},
                     "val": {"ganancia_lista": gv, "envios_lista": nv}, "test": {"ganancia_lista": gt, "envios_lista": nt},
                     "vs_lista": {"val": gv - lista["val"], "test": gt - lista["test"]},
                     "decision": "mejora" if gv > lista["val"] and gt > lista["test"] else "no mejora"})


# ---------------------------------------------------------------- sobre el loop

def h33_h34() -> None:
    from scipy.stats import spearmanr
    pasos = [f for f in csv.DictReader(open(arnes.RAIZ / "v3" / "loop.tsv", encoding="utf-8"), delimiter="\t")
             if f.get("d_auc")]
    pred = np.array([float(f["pred_auc"]) for f in pasos])
    obs = np.array([float(f["d_auc"]) for f in pasos])
    rho = spearmanr(pred, obs)
    acierto = float(np.mean(np.sign(pred) == np.sign(obs)))
    escribir("H33", {"id": "H33", "hipotesis": "H33", "pasos": len(pasos), "spearman": float(rho.statistic),
                     "p": float(rho.pvalue), "acierto_de_signo": acierto,
                     "predijo_suba": int((pred > 0).sum()), "subio": int((obs > 0).sum()),
                     "media_predicha": float(pred.mean()), "media_observada": float(obs.mean()),
                     "decision": "correlación positiva" if rho.statistic > 0 and rho.pvalue < 0.05 else "sin correlación"})
    ahorro, total, perdidos = 0.0, 0.0, []
    for f in pasos:
        paso, vig = leer(f["corrida"]), leer(f["vigente"])
        a1 = paso["por_semilla"]["val"][0]["auc"]
        b1 = vig["por_semilla"]["val"][0]["auc"]
        total += float(f["segundos"])
        if a1 <= b1 - 0.002:
            ahorro += float(f["segundos"]) * (1 - 1 / arnes.SEMILLAS_LOOP)
            if f["decision"] == "queda":
                perdidos.append(f["corrida"])
    escribir("H34", {"id": "H34", "hipotesis": "H34", "segundos_total": total, "segundos_ahorrados": ahorro,
                     "fraccion_ahorrada": ahorro / total if total else 0.0, "cambios_que_se_habrian_perdido": perdidos,
                     "decision": "ahorra sin perder" if ahorro > 0 and not perdidos else
                                 ("pierde un cambio que quedó" if perdidos else "no ahorra")})


# ---------------------------------------------------------------- Holm

def holm_y_decisiones() -> dict:
    ps = {}
    for h, k in CONFIRMATORIAS.items():
        if (R / h / "resultado.json").exists():
            ps[h] = leer(h)["vs_base"][k]["p"]
    aj = arnes.holm(ps)
    for h, p in aj.items():
        d = leer(h)
        k = CONFIRMATORIAS[h]
        v = d["vs_base"][k]
        v["p_holm"] = p
        par = v.get("pareado")
        ok = v["delta"] > 0 and v["ic"][0] > 0 and p < 0.05 and (par is None or par["media"] > 2 * par["se"])
        d["decision"] = "mejora" if ok else "no mejora"
        escribir(h, d)
    return aj


def main() -> int:
    h25()
    h18()
    corte()
    h33_h34()
    for h, p in holm_y_decisiones().items():
        print(f"{h}: p de Holm {p:.4f} → {leer(h)['decision']}")
    for h in ("H36", "H37", "H38", "H33", "H34"):
        print(h, {k: v for k, v in leer(h).items() if k in ("corte", "vs_lista", "spearman", "acierto_de_signo",
                                                               "fraccion_ahorrada", "decision")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
