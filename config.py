"""Configuración fija del experimento: datos, meses, ganancia, semillas y reglas.

El loop de autoresearch NO edita este archivo. Lo que se puede cambiar por variable de entorno es sólo para correr las
pruebas con datos sintéticos (`tools/sinteticos.py`).

Escenario (Segunda Competencia, DMEyF 2026): el modelo se DEFINE sólo con datos de 2019. El presente es 201912: se ven
los datos hasta 201912 y la clase como se sabía entonces (completa hasta 201910, 201911 sólo BAJA+1, 201912 nada).
Cada experimento se mide en 201908, 201909 y 201910 con la misma regla que en producción: se reentrena con los últimos
6 meses y para puntuar m se sabe la clase completa hasta m - 2 y sólo las BAJA+1 de m - 1. En 2020 el modelo elegido
sólo se reentrena (tools/produccion.py lo informa al final); los cambios para 202007-202009 se deciden en el aula.
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

# ---------------------------------------------------------------- train y test (todo dentro de 2019)
PRESENTE = 201912                      # último mes que se ve al definir el modelo
MESES_TEST = [201908, 201909, 201910]  # deciden los experimentos: los tres últimos meses con clase completa en 201912
# Producción (sólo para informar al final, nunca para decidir): el modelo elegido se reentrena mes a mes en 2020.
MESES_PRODUCCION = [202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008, 202009]
ULTIMO_MES_PRODUCCION = 202011         # la clase de 202009 sale de 202010 y 202011

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
