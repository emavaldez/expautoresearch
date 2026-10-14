"""Reserva y test sellado (202108). El loop NO usa esta herramienta: la corre una persona al final.

    python tools/sellado.py reserva E004 E009 E012            # elige entre finalistas (máx. 3) en el fold de reserva
    python tools/sellado.py test E009 [--semillas 20]          # mide un candidato en 202108 (cuenta como lectura)
    python tools/sellado.py test --modulo base/modelo.py --referencia --nombre referencia
    python tools/sellado.py lista base/lista.csv --nombre referencia   # una lista fija de clientes a estimular
    python tools/sellado.py estado

Test: entrena con clase hasta config.ENTREGA["ultima_clase"], puntúa config.MES_TEST y decide cuántos estimular con
la regla CORTE del train.py (por defecto "ratio": el promedio de los envíos óptimos de los folds de selección,
como proporción de los clientes del mes). Las listas con ids quedan en la caché; al repositorio va sólo el resumen.
Los candidatos tienen config.MAX_LECTURAS_TEST lecturas; la referencia no cuenta.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import comun
import config as C
import prepare

RESERVA = comun.RAIZ / "sellado" / "reserva.tsv"
TEST = comun.RAIZ / "sellado" / "test.tsv"
COL_RESERVA = ["fecha", "experimento", "semillas", "ganancia_ensamble", "envios", "elegido"]
COL_TEST = ["fecha", "nombre", "tipo", "semillas", "corte", "envios", "ganancia_corte", "baja2_en_lista",
            "baja2_total", "ganancia_max_suavizada", "envios_max", "lectura"]


def etiquetas_test() -> set[int]:
    """Ids BAJA+2 del mes de test. Acepta: una columna de ids (todos positivos), una columna clase_ternaria, o una
    segunda columna 0/1 o 'BAJA+2'."""
    df = pd.read_csv(C.ETIQUETAS_TEST)
    ids = df[prepare.ID] if prepare.ID in df.columns else df.iloc[:, 0]
    if df.shape[1] == 1:
        return set(ids.astype(np.int64))
    col = prepare.CLASE if prepare.CLASE in df.columns else [c for c in df.columns if c != ids.name][0]
    v = df[col]
    pos = v.astype(str).str.upper().str.replace(" ", "").eq("BAJA+2") | (pd.to_numeric(v, errors="coerce") == 1)
    return set(ids[pos].astype(np.int64))


def _cantidad(mod, id_: str | None, n_clientes: int, prob: np.ndarray, envios: int | None) -> tuple[str, int]:
    if envios:
        return f"fijo {envios} (a mano)", int(envios)
    corte = getattr(mod, "CORTE", {"tipo": "ratio"})
    if corte["tipo"] == "fijo":
        return f"fijo {corte['envios']}", int(corte["envios"])
    if corte["tipo"] == "umbral":
        return f"umbral p>{corte['p']}", int((prob > corte["p"]).sum())
    ratio = comun.resultado(id_)["ratio_envios"] if id_ else None
    if ratio is None:
        raise ValueError("el corte 'ratio' necesita un experimento con resultado.json; usá CORTE fijo o umbral")
    return f"ratio {ratio:.4f}", int(round(ratio * n_clientes))


def _medir_lista(ids_lista: np.ndarray, positivos: set[int]) -> dict:
    tp = int(np.isin(ids_lista, list(positivos)).sum())
    return {"envios": len(ids_lista), "baja2_en_lista": tp, "baja2_total": len(positivos),
            "ganancia_corte": tp * C.GAN_ACIERTO - len(ids_lista) * C.COSTO_ESTIMULO}


def test(a) -> None:
    est = comun.estado()
    if not a.referencia and est.get("lecturas_test", 0) >= C.MAX_LECTURAS_TEST and not a.forzar:
        sys.exit(f"Ya se usaron las {C.MAX_LECTURAS_TEST} lecturas del test. --forzar lo registra como forzada.")
    if a.modulo:
        mod, nombre = comun.cargar_modulo(Path(a.modulo), "modelo_externo"), a.nombre or Path(a.modulo).stem
    else:
        mod, nombre = comun.modulo_experimento(a.experimento), a.nombre or a.experimento
    semillas = prepare.semillas("final", a.semillas)
    ids, prob = prepare.predecir(mod, C.ENTREGA["ultima_clase"], C.MES_TEST, semillas)
    positivos = etiquetas_test()
    regla, n = _cantidad(mod, a.experimento, len(ids), prob, a.envios)
    orden = np.argsort(-prob, kind="stable")
    lista = ids[orden][:n]
    r = _medir_lista(lista, positivos)
    g_max, n_max = prepare.ganancia_suavizada(prob, np.isin(ids, list(positivos)))
    lectura = "referencia" if a.referencia else str(est.get("lecturas_test", 0) + 1) + (" (forzada)" if est.get(
        "lecturas_test", 0) >= C.MAX_LECTURAS_TEST else "")
    (C.CACHE / "sellado").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({prepare.ID: lista}).to_csv(C.CACHE / "sellado" / f"{nombre}.csv", index=False)
    filas = comun.leer_tsv(TEST)
    filas.append({"fecha": comun.ahora(), "nombre": nombre, "tipo": "referencia" if a.referencia else "candidato",
                  "semillas": len(semillas), "corte": regla, **r, "ganancia_max_suavizada": round(g_max),
                  "envios_max": n_max, "lectura": lectura})
    comun.escribir_tsv(TEST, filas, COL_TEST)
    if not a.referencia:
        est["lecturas_test"] = est.get("lecturas_test", 0) + 1
        comun.guardar_estado(est)
    print(f"{nombre}: {comun.millones(r['ganancia_corte'])} con {n:,} envíos ({regla}); {r['baja2_en_lista']} de "
          f"{r['baja2_total']} BAJA+2 | máximo suavizado (con la clase a la vista): {comun.millones(g_max)} en {n_max:,}")
    _cerrar(f"test sellado: {nombre}")


def lista(a) -> None:
    ids = pd.read_csv(a.archivo).iloc[:, 0].astype(np.int64).to_numpy()
    r = _medir_lista(ids, etiquetas_test())
    filas = comun.leer_tsv(TEST)
    filas.append({"fecha": comun.ahora(), "nombre": a.nombre, "tipo": "referencia", "semillas": "",
                  "corte": "lista fija", **r, "ganancia_max_suavizada": "", "envios_max": "", "lectura": "referencia"})
    comun.escribir_tsv(TEST, filas, COL_TEST)
    print(f"{a.nombre}: {comun.millones(r['ganancia_corte'])} con {r['envios']:,} envíos; "
          f"{r['baja2_en_lista']} de {r['baja2_total']} BAJA+2")
    _cerrar(f"test sellado: lista {a.nombre}")


def reserva(a) -> None:
    if len(a.experimentos) > 3:
        sys.exit("Como máximo 3 finalistas")
    if comun.leer_tsv(RESERVA) and not a.forzar:
        sys.exit("La reserva ya se usó. --forzar la vuelve a leer (queda registrado).")
    semillas = prepare.semillas("final", a.semillas)
    df = prepare.cargar()
    filas, mejores = [], []
    for e in a.experimentos:
        r = prepare.evaluar(comun.modulo_experimento(e), C.FOLDS_RESERVA, semillas, df=df)
        g = prepare.media_ensamble(r, C.FOLDS_RESERVA)
        mejores.append((g, e))
        filas.append({"fecha": comun.ahora(), "experimento": e, "semillas": len(semillas), "ganancia_ensamble": round(g),
                      "envios": r[str(C.FOLDS_RESERVA[0])]["ensamble"]["envios"], "elegido": ""})
    ganador = max(mejores)[1]
    for f in filas:
        f["elegido"] = "sí" if f["experimento"] == ganador else ""
    comun.escribir_tsv(RESERVA, comun.leer_tsv(RESERVA) + filas, COL_RESERVA)
    for g, e in sorted(mejores, reverse=True):
        print(f"{e}: {comun.millones(g)}")
    print(f"elegido por la reserva: {ganador}")
    _cerrar(f"reserva: elegido {ganador}")


def _cerrar(mensaje: str) -> None:
    import pagina
    pagina.generar()
    comun.commit(mensaje, [comun.RAIZ / "sellado", comun.ESTADO, comun.RAIZ / "docs"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reserva")
    r.add_argument("experimentos", nargs="+")
    r.add_argument("--semillas", type=int, default=20)
    r.add_argument("--forzar", action="store_true")
    t = sub.add_parser("test")
    t.add_argument("experimento", nargs="?")
    t.add_argument("--modulo")
    t.add_argument("--nombre")
    t.add_argument("--semillas", type=int, default=20)
    t.add_argument("--envios", type=int, help="cantidad fija de estímulos (si no, la regla CORTE del modelo)")
    t.add_argument("--referencia", action="store_true", help="modelo de referencia: no cuenta como lectura")
    t.add_argument("--forzar", action="store_true")
    li = sub.add_parser("lista")
    li.add_argument("archivo")
    li.add_argument("--nombre", default="referencia")
    sub.add_parser("estado")
    a = ap.parse_args()
    if a.cmd == "reserva":
        reserva(a)
    elif a.cmd == "test":
        if not a.experimento and not a.modulo:
            ap.error("indicá un experimento o --modulo")
        test(a)
    elif a.cmd == "lista":
        lista(a)
    else:
        e = comun.estado()
        print(f"lecturas del test usadas: {e.get('lecturas_test', 0)} de {C.MAX_LECTURAS_TEST}")
        for f in comun.leer_tsv(TEST):
            print(f"{f['fecha']} {f['nombre']} ({f['tipo']}): {comun.millones(f['ganancia_corte'])} con {f['envios']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
