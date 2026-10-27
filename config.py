"""Configuración fija del experimento: datos, meses, ganancia, semillas y reglas.

El loop de autoresearch NO edita este archivo. Lo que se puede cambiar por variable de entorno es sólo para correr las
pruebas con datos sintéticos (`tools/sinteticos.py`).

Escenario (Segunda Competencia, DMEyF 2026): el modelo nace con datos de 2019 y se reentrena todos los meses con los
últimos 6 meses. Para puntuar un mes m se sabe la clase completa hasta m - 2 y sólo las BAJA+1 de m - 1. Se mide en
202007, 202008 y 202009, los tres meses que en el aula llegan de a uno (con la data modificada).
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# ---------------------------------------------------------------- datos
# Dataset de la Segunda Competencia (201901 a 202108; csv, csv.gz o parquet), una fila por cliente y mes.
DATOS = Path(os.environ.get("EXP_DATOS", "~/competencia2/datos/competencia_02_crudo.parquet")).expanduser()
# Caché local (parquet, probabilidades, listas con ids). Nunca va al repositorio.
CACHE = Path(os.environ.get("EXP_CACHE", "~/.cache/expautoresearch")).expanduser()

# ---------------------------------------------------------------- problema
GAN_ACIERTO = 1_072_500      # por cada BAJA+2 estimulado
COSTO_ESTIMULO = 27_500      # por cada estímulo
VENTANA_SUAVIZADO = 401      # envíos: la meseta de la curva, para leer la calidad del orden sin depender del corte
GAP = 2                      # para puntuar m se conoce la clase completa hasta m - GAP
VENTANA = 6                  # regla del escenario: se entrena con los meses m - 5 a m - 1 (m - 1 sólo con sus BAJA+1)

# ---------------------------------------------------------------- train y test
MESES_TEST = [202007, 202008, 202009]  # deciden los experimentos: la ganancia real de la lista de cada mes
ULTIMO_MES_DATOS = 202011              # la clase de 202009 sale de 202010 y 202011; nada más allá se lee

# ---------------------------------------------------------------- semillas
# Semilla primigenia propia: las semillas son los primos que le siguen.
SEMILLA_PRIMIGENIA = int(os.environ.get("EXP_SEMILLA", "800011"))
# Bloques que no se mezclan: las de screening "ven" muchas decisiones, las de confirmación no.
BLOQUES = {
    "screening": (0, 5),       # cada corrida de run.py
    "confirmacion": (5, 25),   # tools/comparar.py
    "busqueda": (25, 45),      # tools/hpo.py
    "final": (45, 145),
}

# ---------------------------------------------------------------- reglas
PRESUPUESTO_S = int(os.environ.get("EXP_PRESUPUESTO", "14400"))  # tope de una corrida de run.py (se corta al 150%)
UMBRAL_SCREENING = 3_000_000   # mejora mínima de la ganancia media por mes de test para pasar a confirmación
MIN_MESES_GANADOS = 2          # y tiene que mejorar en al menos esta cantidad de meses de test

REPO_GITHUB = "emavaldez/expautoresearch"
