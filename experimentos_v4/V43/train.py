"""
v4 · el modelo de la primera competencia (E019) con un interruptor por idea en CAMBIOS (ver v4/program.md y la sección
v4 de PROTOCOLO.md). Con CAMBIOS en sus valores por defecto es exactamente E019. Lo que sigue es el texto de E019.

B0 · modelo base: "Tips Ale Bolaño" de la Primera Competencia (github.com/emavaldez/primeracompetencia), llevado a la
Segunda Competencia. Lo único que el loop modifica es este archivo.

Las tres piezas son las mismas que en la entrega de la primera:
  1. GENERAL: LightGBM multiclase (CONTINUA / BAJA+1 / BAJA+2) con los tres últimos meses con clase, submuestreo 0,2 de los
     CONTINUA, BAJA+1 a medio peso, variables crudas + lags y deltas de 1 y 2 meses + razones y ventana de 3 meses, el
     percentil con cero fijo de 14 montos, fultimo_cierre menos la moda del mes y el FE histórico de la cátedra
     (tendencia y promedio de 3 meses).
  2. ESPECIALISTA del grupo chico y riesgoso según internet, binario, sólo con el último mes con clase y con las
     variables que les ganan a los canaritos; se combina con el general por reparto.
  3. SEGUNDA ETAPA: LightGBM chico que reordena la franja dudosa del resto de los clientes, entrenado con los dos
     últimos meses con clase puntuados por una base que no vio su clase.

Qué cambió para estos datos (2019 en lugar de 2021), sin tocar la lógica:
  - ROTAS: la tabla de variables rotas por mes es la de 2019 (201904, 201905 y 201910, y las de transferencias y app
    de la primera mitad del año, que todavía no existían). En la primera era sólo ccajas_depositos en 202105. Lo que se
    rompa en 2020 la tabla no lo sabe (ver P1).
  - internet: en 2021 marcaba deudores con 1 a 3 (4% de los clientes). En 2019 es binaria y al revés: 1 = usa
    homebanking (92%, 0,3% de BAJA+2) y 0 es el grupo chico y riesgoso (8%, 2,1 a 2,4% de BAJA+2 en 2019). El especialista se
    arma sobre internet == 0 y la segunda etapa sobre internet == 1, que es el "resto" equivalente.
  - FRANJA de la segunda etapa: 8.000 a 30.000 puestos rodeaba un corte de 11.250 sobre 164.000 clientes; acá el corte es
    de unos 4.400 sobre 133.000 (2019), así que la franja se achica en la misma proporción: 3.100 a 11.500.
  - CORTE: 3,3% de los clientes del mes (la proporción que maximizaba la ganancia en la validación de 2019); la primera
    usaba 11.250 fijos.
  - Todo se definió con datos de 2019 (ver config.py). Ventana: para puntuar m entrena con los meses con clase completa
    de m - 5 a m - 2 (regla del escenario); el general
    usa los tres últimos y la segunda etapa los dos últimos, puntuados por una base entrenada con los anteriores.
"""
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- features

# Empiezan con c pero no son conteos: edades (crecen solas: no llevan razones ni ventana; sí lags, deltas y el FE de la
# cátedra) y banderas 0/1.
NO_MAGNITUD = {
    "cliente_edad", "cliente_antiguedad",
    "cliente_vip", "cdescubierto_preacordado", "ccajas_transacciones", "cmobile_app_trx",
}


def es_magnitud(c: str) -> bool:
    """Montos (m*) y conteos (c*), del cliente o de sus tarjetas (Visa_m*, Master_c*): 128 de las 152 variables.

    Quedan afuera las fechas (F*, f*), las banderas y los códigos (t*, status, delinquency, internet, active_quarter):
    ahí una razón o un promedio no significan nada.
    """
    nombre = c.split("_", 1)[1] if c.startswith(("Visa_", "Master_")) else c
    return nombre[0] in "mc" and c not in NO_MAGNITUD


def razon(num, den):
    """num / den; con denominador 0 da NaN (nunca infinito)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, np.nan, num / den)


# Data Quality (tips de la cátedra): una variable rota en un mes se pasa a NA sólo en ese mes, antes de lags y deltas.
# Tabla fija, medida con los datos de 2019: la variable está casi toda en cero o vacía ese mes y en los demás no.
ROTAS = {
    "ccomisiones_otras": (201905, 201910,),
    "chomebanking_transacciones": (201910,),
    "cmobile_app_trx": (201901, 201902, 201903, 201904, 201905, 201906,),
    "ctarjeta_visa_debitos_automaticos": (201904,),
    "ctransferencias_recibidas": (201901, 201902, 201903, 201904, 201905,),
    "mactivos_margen": (201905, 201910,),
    "mcomisiones": (201905, 201910,),
    "mcomisiones_otras": (201905, 201910,),
    "mpasivos_margen": (201905, 201910,),
    "mrentabilidad": (201905, 201910,),
    "mrentabilidad_annual": (201905, 201910,),
    "mtransferencias_recibidas": (201901, 201902, 201903, 201904, 201905,),
    "mttarjeta_visa_debitos_automaticos": (201904,),
    "tmobile_app": (201901, 201902, 201903, 201904, 201905, 201906,),
}

# Data Drifting. Mirando las 152 variables mes a mes, la escala de estos montos se corre de un mes a
# otro: aguinaldo (202106: mpayroll x1,56; el saldo y la caja de ahorro quedan inflados en 202106-07, con el primer cuartil
# del saldo x5) e inflación en tarjeta, comisiones y transferencias (+15 a +55% en 202106-08 contra 202103-05). El modelo
# aprende con 202104-06 y agosto se puntúa con lags de junio y julio: un corte en pesos aprendido en abril no vale lo mismo
# en agosto, y "cayó respecto de junio" en agosto es la resaca del aguinaldo, no una fuga.
# Selección en dos pasos: (1) montos (m*) con algún cuartil de 202106-08 a más de 25% del promedio de 202103-05 y al menos
# 0,5% de la ganancia del modelo (la importancia "gain" de LightGBM); (2) de esos, sólo los que con el percentil quedan MÁS
# parecidos entre el entrenamiento (202104-06) y agosto (PSI de sus columnas, ponderado por esa ganancia: 0,057 -> 0,010
# en estos 14).
# Quedaron afuera los que el percentil empeora, que son los de pocos valores distintos o con muchos empates:
# cpayroll_trx (1, 2, 3: el percentil de "1" pasa de 30 a 15 en junio), mcomisiones_mantenimiento (lista de precios),
# mcaja_ahorro_dolares, Visa_mpagominimo, Visa_mpagospesos y mttarjeta_visa_debitos_automaticos. Los préstamos no derivan.
PERCENTIL_CERO_FIJO = [
    "mcuentas_saldo", "mpayroll", "mcaja_ahorro", "mpasivos_margen", "mtarjeta_visa_consumo", "mcuenta_corriente",
    "mtransferencias_recibidas", "mrentabilidad", "mcomisiones_otras", "mcomisiones", "Visa_msaldototal",
    "Visa_msaldopesos", "mautoservicio", "mtransferencias_emitidas",
]
# Corrimiento de calendario: son "días desde el último cierre" y el grupo mayoritario (69% de los clientes) vale
# 1 / 2 / 5 / 0 / 3 / 6 en 202103..08; el segundo grupo va 7 días atrás. Se le resta la moda del mes: queda el grupo de
# cierre del cliente (0, 7, ...), igual todos los meses. (El percentil no sirve acá: da 36,18 / 36,22 / 36,27... y esos
# decimales le dejan al árbol separar los meses.)
RESTAR_MODA = ["Visa_fultimo_cierre", "Master_fultimo_cierre"]


def percentil_cero_fijo(x: pd.Series, mes: pd.Series) -> pd.Series:
    """Percentil (0 a 100) dentro del mes: usa sólo las filas de ese mes, así que no mira ni adelante ni atrás.

    Los positivos se ordenan entre sí (0, 100], los negativos entre sí [-100, 0) y el cero queda en 0: se conservan el signo
    y el "no tiene", que es lo que el ranking simple borra.
    La escala 0-100 es para que el redondeo a centavos de la ventana no le coma resolución.
    """
    pos = x.where(x > 0).groupby(mes).rank(pct=True)
    neg = (-x).where(x < 0).groupby(mes).rank(pct=True)
    return (100 * (pos.fillna(0) - neg.fillna(0))).where(x.notna())


def montos_sin_percentil(df: pd.DataFrame) -> list:
    """v4: los montos (m*, Visa_m*, Master_m*) que E019 deja en pesos."""
    return [c for c in df.columns if es_magnitud(c) and c not in PERCENTIL_CERO_FIJO
            and (c.split("_", 1)[1] if c.startswith(("Visa_", "Master_")) else c)[0] == "m"]


def lags_y_deltas(df: pd.DataFrame) -> pd.DataFrame:
    order = df.sort_values(["numero_de_cliente", "foto_mes"]).index
    df = df.loc[order]
    for c, meses in ROTAS.items():
        df[c] = df[c].where(~df.foto_mes.isin(meses))
    for c in PERCENTIL_CERO_FIJO + (montos_sin_percentil(df) if CAMBIOS["percentil_montos"] else []):
        df[c] = percentil_cero_fijo(df[c], df.foto_mes)
    for c in RESTAR_MODA:
        df[c] = df[c] - df.groupby("foto_mes")[c].transform(lambda v: v.mode().iloc[0])
    base = [c for c in df.columns if c not in ("numero_de_cliente", "foto_mes")]
    g = df.groupby("numero_de_cliente")[base]
    new = {}
    for k in (1, 2):  # Feature Engineering histórico de la cátedra: lags y delta lags de orden 1 y 2
        lag = g.shift(k)
        new.update({f"{c}_lag{k}": lag[c] for c in base})
        new.update({f"{c}_delta{k}": df[c] - lag[c] for c in base})
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df.sort_index()


def historia(x, l1, l2):
    """FE histórico de la cátedra para una variable, sobre la ventana que dejan 6 meses de datos: el mes actual (x),
    lag1 y lag2, con los que haya."""
    v = np.stack([x, l1, l2])
    # los montos traen centavos: se redondea para que lo que da 0 sea 0 exacto y no un resto de punto flotante
    prom = razon(np.nansum(v, axis=0).round(2), (~np.isnan(v)).sum(axis=0))
    maxi = np.fmax.reduce(v)  # fmax y fmin saltean los NaN
    return {
        # razones sin escala: un árbol no las arma solo y no se corren con la inflación
        "ratiolag1": razon(x, l1),
        "ratioavg3": razon(x, prom),
        "avg3": prom,
        "min3": np.fmin.reduce(v),
        "max3": maxi,
        "ratiomax3": razon(x, maxi),
    }


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe todas las filas de meses <= mes a puntuar (sin clase). Una fila de salida por fila de entrada."""
    magnitudes = [c for c in df.columns if es_magnitud(c)]
    agregar_combinadas(df)
    df = lags_y_deltas(df)
    resumenes = apagadas_y_bajas(df, magnitudes)
    if resumenes:
        df = pd.concat([df, pd.DataFrame(resumenes, index=df.index)], axis=1)
    # Se calcula fila a fila a partir de los lags: sólo mira hacia atrás. Las columnas nuevas no llevan lag ni delta.
    # float32: son 768 columnas más sobre ~980.000 filas y al árbol le sobran 7 dígitos.
    hist = {}
    for c in magnitudes:
        for k, a in historia(*(df[n].to_numpy(float) for n in (c, f"{c}_lag1", f"{c}_lag2"))).items():
            hist[f"{c}_{k}"] = a.astype("float32")
    df = pd.concat([df, pd.DataFrame(hist, index=df.index)], axis=1)
    return df if CAMBIOS["sin_fe_catedra"] else fe_historico_catedra(df)


# ---------------------------------------------------------------- v4: variables propuestas por el agente en v3 (H28)

def _suma(df, cols):
    return df[list(cols)].sum(axis=1, min_count=1).to_numpy(np.float32)


def agregar_combinadas(df: pd.DataFrame) -> None:
    """a, b, e, f: columnas nuevas del mes, antes de la historia (así llevan lags y deltas como las crudas). Ninguna
    empieza con m o c, así que no llevan razones ni ventana."""
    letras = CAMBIOS["agente"]
    if not set(letras) & set("abef"):
        return
    nuevas = {}
    if "e" in letras:  # antes de sumar columnas, para contar sólo las crudas
        nuevas["n_faltantes"] = df[[c for c in df.columns if c not in ("numero_de_cliente", "foto_mes")]].isna().sum(
            axis=1).to_numpy(np.float32)
    tot = {
        "tc_consumo": _suma(df, ["mtarjeta_visa_consumo", "mtarjeta_master_consumo"]),
        "tc_transacciones": _suma(df, ["ctarjeta_visa_transacciones", "ctarjeta_master_transacciones"]),
        "tc_limite": _suma(df, ["Visa_mlimitecompra", "Master_mlimitecompra"]),
        "tc_saldo": _suma(df, ["Visa_msaldototal", "Master_msaldototal"]),
        "tc_pagado": _suma(df, ["Visa_mpagado", "Master_mpagado"]),
        "tc_minimo": _suma(df, ["Visa_mpagominimo", "Master_mpagominimo"]),
        "prestamos_total": _suma(df, ["mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios"]),
        "inversiones_total": _suma(df, ["mplazo_fijo_dolares", "mplazo_fijo_pesos", "minversion1_pesos",
                                        "minversion1_dolares", "minversion2"]),
        "transacciones_total": _suma(df, ["ctarjeta_debito_transacciones", "ctarjeta_visa_transacciones",
                                          "ctarjeta_master_transacciones", "chomebanking_transacciones",
                                          "ccajas_transacciones", "catm_trx", "catm_trx_other", "cmobile_app_trx"]),
    }
    if "a" in letras:
        nuevas.update(tot)
    if "b" in letras:
        prod = df["cproductos"].to_numpy(np.float32)
        edad = df["cliente_edad"].to_numpy(np.float32)
        nuevas.update({
            "tc_uso_limite": razon(tot["tc_saldo"], tot["tc_limite"]),
            "tc_pago_sobre_saldo": razon(tot["tc_pagado"], tot["tc_saldo"]),
            "saldo_por_producto": razon(df["mcuentas_saldo"].to_numpy(np.float32), prod),
            "rentabilidad_por_producto": razon(df["mrentabilidad"].to_numpy(np.float32), prod),
            "antiguedad_sobre_edad": razon(df["cliente_antiguedad"].to_numpy(np.float32), edad * 12),
        })
    if "f" in letras:
        digital = _suma(df, ["chomebanking_transacciones", "cmobile_app_trx"])
        nuevas["parte_digital"] = razon(digital, tot["transacciones_total"])
    for c, v in nuevas.items():
        df[c] = v


def apagadas_y_bajas(X: pd.DataFrame, magnitudes: list[str]) -> dict:
    """c y d: resúmenes por cliente sobre todos los montos y conteos (el árbol no puede contar columnas)."""
    letras = CAMBIOS["agente"]
    if not set(letras) & set("cd"):
        return {}
    M = X[magnitudes].to_numpy(np.float32)
    L1 = X[[f"{c}_lag1" for c in magnitudes]].to_numpy(np.float32)
    L2 = X[[f"{c}_lag2" for c in magnitudes]].to_numpy(np.float32)
    out = {}
    if "c" in letras:
        out["n_apagadas1"] = ((L1 > 0) & (M == 0)).sum(axis=1).astype(np.float32)
        out["n_apagadas2"] = ((L2 > 0) & (M == 0)).sum(axis=1).astype(np.float32)
        out["n_en_cero"] = (M == 0).sum(axis=1).astype(np.float32)
    if "d" in letras:
        for k, L in ((1, L1), (2, L2)):
            ok = ~np.isnan(M) & ~np.isnan(L)
            out[f"frac_bajaron{k}"] = razon(((M < L) & ok).sum(axis=1).astype(np.float32), ok.sum(axis=1).astype(np.float32))
    return out


# ---------------------------------------------------------------- FE histórico de la cátedra: tendencia y promedio de 3 meses
#
# Es lo que agregó la cátedra a su Feature Engineering Histórico (función TendenciaYmuchomas con ventana 3, tendencia y
# promedio): la pendiente por cuadrados mínimos de los últimos tres valores y su promedio, para todas las variables con lag.
# Con los tres meses la pendiente es (x - lag2) / 2; con dos, x - lag1; con uno queda vacía (igual que en su código).
# Quedan afuera el detalle del resumen de tarjeta (Visa_*, Master_*: su historia no entra, ver historia_de_resumen) e internet
# (tiene su propio tratamiento). El promedio sólo se agrega a las variables que no lo tenían (las que no son montos ni conteos).
# Estas columnas las ve sólo el modelo general.
FE_CATEDRA = ("_tend3", "_avg3p")


def fe_historico_catedra(d: pd.DataFrame) -> pd.DataFrame:
    crudas = [c for c in d.columns if f"{c}_lag1" in d.columns and f"{c}_lag2" in d.columns
              and not c.startswith(("Visa_", "Master_")) and c != "internet"]
    nuevo = {}
    for c in crudas:
        y = d[[f"{c}_lag2", f"{c}_lag1", c]].to_numpy(float)
        x = np.array([1.0, 2.0, 3.0]); ok = ~np.isnan(y); n = ok.sum(1); y0 = np.where(ok, y, 0)
        sx, sy, sxy, sxx = (ok * x).sum(1), y0.sum(1), (y0 * x).sum(1), (ok * x * x).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            a = sy / n
        t[n < 2] = np.nan; a[n < 2] = np.nan
        nuevo[f"{c}_tend3"] = t.astype("float32")
        if f"{c}_avg3" not in d.columns:
            nuevo[f"{c}_avg3p"] = a.astype("float32")
    return pd.concat([d, pd.DataFrame(nuevo, index=d.index)], axis=1)


# ---------------------------------------------------------------- modelo

TRAIN_MONTHS_BACK = 4  # E1
MESES_ENTRENAMIENTO = 4  # la ventana entera: m - 5 a m - 2 (la segunda etapa necesita los dos primeros para su base)
HISTORIA_MESES = 2       # lags de 1 y 2 meses
CORTE = {"tipo": "fraccion", "f": 0.033}

PARAMS = dict(
    objective="binary",  # sin efecto: base() lo pisa con "multiclass" de 3 clases
    learning_rate=0.027,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.5,
    max_bin=255,
    verbosity=-1,
    num_threads=16,
)
NUM_ROUNDS = 500
UNDERSAMPLING = 0.2  # fracción de CONTINUA que se conserva
POSITIVOS = ["BAJA+1", "BAJA+2"]  # las dos clases de baja
PESO_BAJA1 = 0.5  # las BAJA+1 pesan la mitad al entrenar el general: se van en un mes y no pagan; BAJA+2 es la clase que paga
PESO_BAJA2 = 1.0  # v4: peso de las BAJA+2 (lo mueve la búsqueda)

# v4: un interruptor por idea (v4/cola.tsv). Los valores de acá son E019; el loop cambia uno por paso.
CAMBIOS = {
    "franja": 1,                    # V1-V10: enfoque para reordenar la franja dudosa (0 = no se toca)
    "puntaje_baja1": 1.0,           # puntaje = P(BAJA+2) + a · P(BAJA+1)
    "binaria": False,               # BAJA+1 o BAJA+2 contra CONTINUA, en lugar de multiclase
    "historia_resumen": False,      # la historia (lags, deltas, razones) del resumen de tarjetas entra
    "sin_fe_catedra": True,        # sin tendencia y promedio de 3 meses de la cátedra
    "sin_enmascarar_internet": False,
    "canaritos_general": False,     # el general usa sólo las variables que les ganan a los canaritos
    "xgboost": False,               # ensamble con un XGBoost con los mismos datos
    "catedra": False,               # ensamble con un modelo al estilo B0 + H1b (binaria, sin submuestreo)
    "params": {'num_leaves': 254, 'learning_rate': 0.0158, 'num_rounds': 667, 'min_data_in_leaf': 72, 'feature_fraction': 0.517, 'lambda_l2': 0.0033, 'submuestreo': 0.132},                 # hiperparámetros de una búsqueda (v4/hpo.py): ver aplicar()
    "franja_params": {'franja_desde': 0.5600014111274607, 'franja_hasta': 3.5553427460608167, 'learning_rate': 0.07447262553981542, 'num_leaves': 7, 'min_data_in_leaf': 599, 'feature_fraction': 0.2705345015055787, 'lambda_l2': 89.53643923682695, 'rondas': 176, 'peso_base': 1.0477883025328396},          # parámetros de la franja de una búsqueda (v4/hpo_franja.py): ver aplicar_franja()
    "agente": "",                   # variables del agente de v3 (H28): letras de "abcdef"
    "binaria_ensamble": False,      # ensamble con la binaria BAJA+1 o BAJA+2 (E027)
    "percentil_montos": False,      # todos los montos a percentil dentro del mes (E019: sólo 14)
    "peso_mes": 1.0,                # cada mes de entrenamiento pesa r^(meses antes del último)
    "peso_miembros": 1.0,           # peso de cada miembro en el promedio de puestos (el general pesa 1)
    "sacar": ('_lag1', '_lag2'),                    # sufijos de las familias de variables que no ve ningún modelo
    "miembros": [],                 # el general con otras constantes (ver miembro()), cada uno un miembro más del ensamble
}


HISTORIA = tuple(f"_{s}" for s in ("lag1", "delta1", "lag2", "delta2", "ratiolag1", "ratioavg3", "avg3", "min3", "max3", "ratiomax3"))


def historia_de_resumen(c: str) -> bool:
    """Historia derivada (lags, deltas, razones, ventana) del detalle del resumen de tarjeta (Visa_*, Master_*): no entra.

    Esas 44 variables son la foto del último resumen y dependen de en qué día cae el cierre (fultimo_cierre), que se
    corre con el calendario: es el bloque que más cambia de distribución entre los meses de entrenamiento y el mes a
    puntuar (KS mediano de sus derivadas 0,037 contra 0,016 del resto; las fechas llegan a 0,7-1,0). La diferencia de
    un saldo de resumen contra el mes anterior mezcla el ciclo del resumen con lo que hizo el cliente. Entran crudas
    (el modelo sigue viendo saldo, pago mínimo, mora y límite de hoy), y la historia del uso de la tarjeta queda en los
    agregados del banco (ctarjeta_*_transacciones, mtarjeta_*_consumo), que sí conservan sus derivadas.
    Junto con el tratamiento de la deriva: Visa_msaldototal y Visa_msaldopesos (a percentil dentro del mes) y los dos
    fultimo_cierre (menos la moda del mes) son de este bloque, así que entran sólo con su valor del mes ya corregido;
    de los 14 montos a percentil, los otros 12 son del banco y conservan su historia, calculada sobre el percentil.
    """
    return not CAMBIOS["historia_resumen"] and c.startswith(("Visa_", "Master_")) and c.endswith(HISTORIA)

# ---------------------------------------------------------------- especialista de deudores, con canaritos
#
# Por qué existe el especialista: `internet` con valor 1 a 3 marca deudores en problemas (saldo negativo, mora de tarjeta, 3 o más
# préstamos). Sus bajas son cierres que decide el banco y cambian de un mes a otro (2,3% de BAJA+2 en 202104, 1,6% en 202105 y
# 4,3% en 202106); el general promedia los tres meses y los subestima. El especialista mira SÓLO a ese grupo y SÓLO el último
# mes con clase. Por qué con canaritos: con ~6.000 filas y ~380 bajas, 1.160
# variables son demasiadas y el especialista ajusta ruido. Los canaritos miden cuánta ganancia junta una columna de números
# al azar; sólo quedan las variables reales que juntan más que el mejor de ellos.
SEGMENTO_INTERNET = 0  # en 2019 el grupo chico y riesgoso es internet == 0 (ver el encabezado)
RESTO_INTERNET = 1     # el resto, sobre el que actúa la segunda etapa
PARAMS_ESP = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,  # el segmento son ~6.000 filas y ~380 bajas: árboles chicos
    min_data_in_leaf=50,
    feature_fraction=0.3,
    max_bin=255,
    verbosity=-1,
    num_threads=4,  # con tan pocas filas más hilos no aceleran
)
NUM_ROUNDS_ESP = 200
MIN_FILAS_ESP = 200  # resguardo: con menos filas o una sola clase no hay nada que aprender
CANARITOS = 50    # cuántas columnas de ruido
REPETICIONES = 5  # corridas con canaritos que se promedian
MINIMO = 5        # nunca menos variables que éstas
_ELEGIDAS = {}    # train del segmento -> lista: la selección no depende de la semilla, se calcula una vez por proceso


def le_ganan_al_ruido(tr: pd.DataFrame, cols: list, y: np.ndarray) -> list:
    """Variables de cols cuya ganancia media supera a la del mejor canarito, ordenadas de mayor a menor ganancia."""
    mes = int(tr.foto_mes.iloc[0])
    clave = (mes, len(tr), int(tr.numero_de_cliente.sum()), int(y.sum()), tuple(cols))  # el train exacto y sus columnas
    with _CANDADO_CACHE:
        return _le_ganan_al_ruido(tr, cols, y, mes, clave)


def _le_ganan_al_ruido(tr, cols, y, mes, clave):
    if clave not in _ELEGIDAS:
        X = tr[cols].reset_index(drop=True)
        ruido = [f"canarito_{i}" for i in range(CANARITOS)]
        ganancias = []
        for j in range(REPETICIONES):
            rng = np.random.default_rng(10_000 * CANARITOS + 100 * (mes % 100) + j)
            D = pd.concat([X, pd.DataFrame(rng.random((len(X), CANARITOS)), columns=ruido)], axis=1)
            model = lgb.train({**PARAMS_ESP, "seed": 9000 + j}, lgb.Dataset(D, label=y), num_boost_round=NUM_ROUNDS_ESP)
            ganancias.append(model.feature_importance("gain"))
        g = np.mean(ganancias, axis=0)
        real, vara = g[:len(cols)], g[len(cols):].max()
        orden = np.argsort(-real, kind="stable")
        elegidas = [cols[i] for i in orden if real[i] > vara]
        _ELEGIDAS[clave] = elegidas if len(elegidas) >= MINIMO else [cols[i] for i in orden[:MINIMO]]
        # Para diagnóstico: el conjunto y el ORDEN de estas variables son parte del modelo (feature_fraction sortea por
        # posición). Si en otra máquina la lista final no coincide, lo primero a mirar es si salen las mismas.
        print(f"  especialista de deudores ({mes}): {len(_ELEGIDAS[clave])} de {len(cols)} variables le ganan al ruido: "
              + ", ".join(_ELEGIDAS[clave]), flush=True)
    return _ELEGIDAS[clave]


def especialista(train: pd.DataFrame, score: pd.DataFrame, cols: list, seed: int):
    """Probabilidad de baja (BAJA+1 o BAJA+2) de los clientes del segmento, en la escala del general.

    Devuelve (máscara de filas de score que son del segmento, probabilidad de esas filas) o (máscara, None) si no se pudo
    entrenar. Se entrena sin submuestreo y se corrige a la escala del general, p / (p + 0,2 (1 - p)), sólo
    con las variables que le ganan al ruido.
    """
    en_seg = (score.internet == SEGMENTO_INTERNET).to_numpy()
    ultimo = train.foto_mes.max()
    tr = train[(train.foto_mes == ultimo).to_numpy() & (train.internet == SEGMENTO_INTERNET).to_numpy()]
    y = (tr.clase_ternaria != "CONTINUA").to_numpy().astype(int)
    if len(tr) < MIN_FILAS_ESP or y.min() == y.max() or not en_seg.any():
        return en_seg, None
    cols_e = [c for c in cols if c != "foto_mes"]  # un solo mes: foto_mes es constante
    elegidas = le_ganan_al_ruido(tr, cols_e, y)
    model = lgb.train({**PARAMS_ESP, "seed": int(seed)}, lgb.Dataset(tr[elegidas], label=y), num_boost_round=NUM_ROUNDS_ESP)
    p = model.predict(score.loc[en_seg, elegidas])
    return en_seg, p / (p + UNDERSAMPLING * (1 - p))


def preparar(train: pd.DataFrame, seed: int, peso_fila=None, objetivo=None):
    """Lo que ve el general: (filas, columnas, clase, pesos, objetivo). v4: lo usan también los modelos del ensamble."""
    objetivo = objetivo or ("binaria" if CAMBIOS["binaria"] else "multiclase")
    months = sorted(train.foto_mes.unique())[-TRAIN_MONTHS_BACK:]
    train = train[train.foto_mes.isin(months)]
    # Columnas: todas menos la historia del resumen de tarjetas (ver historia_de_resumen); el FE de la cátedra lo ve sólo el general.
    cols = _cols(train)  # P3: sin foto_mes
    # El especialista va antes del submuestreo (usa todas las filas del segmento) y antes de enmascarar internet:
    # igual sólo mira el último mes con clase, que el enmascarado no toca.
    elegidas = general_canaritos(train, cols) if CAMBIOS["canaritos_general"] else None
    # modelo general, para todos los clientes: multiclase, con submuestreo de los CONTINUA y las BAJA+1 a medio peso
    keep = np.random.default_rng(seed).random(len(train)) < UNDERSAMPLING
    train = train[keep | (train.clase_ternaria != "CONTINUA").to_numpy()]
    if objetivo == "binaria":
        y = (train.clase_ternaria != "CONTINUA").astype(int)
        w = np.ones(len(train))
    else:
        y = train.clase_ternaria.map({"CONTINUA": 0, "BAJA+1": 1, "BAJA+2": 2})
        w = np.select([train.clase_ternaria == "BAJA+1", train.clase_ternaria == "BAJA+2"], [PESO_BAJA1, PESO_BAJA2], 1.0)
    if peso_fila is not None:  # enfoque 5
        w = w * peso_fila.reindex(train.index).fillna(1.0).to_numpy()
    if CAMBIOS["peso_mes"] != 1.0:  # v4: r^(meses antes del último mes de entrenamiento)
        atras = len(months) - 1 - np.searchsorted(np.asarray(months), train.foto_mes.to_numpy())
        w = w * CAMBIOS["peso_mes"] ** atras
    # internet cambió de régimen: a igual actividad, tener internet >= 2 dos meses seguidos no subía la baja en 202104-05
    # (1-3% contra 2-6% del resto) y en 202106 la duplica (6-7% contra 2-4%). El promedio de los tres meses borra lo que
    # vale hoy, así que su efecto se aprende SÓLO del último mes con clase: en los meses anteriores de entrenamiento va
    # en el valor de la mayoría. No en NaN: "internet falta" marcaría el mes y el árbol aprendería un efecto de junio.
    # Va junto con la quita de la historia del resumen porque no se pisan: aquel filtro sólo mira Visa_*/Master_* y acá
    # quedan las 5 columnas de internet (cruda, 2 lags, 2 deltas). El assert es para que un cambio de nombres no deje
    # el enmascarado vacío sin avisar (el modelo entrenaría igual, con internet promediado en los tres meses).
    internet = [c for c in train.columns if c.startswith("internet")]
    assert len(internet) == 5, f"se esperaban 5 columnas de internet y hay {internet}"
    # En 2019 el valor mayoritario es 1 (no 0, como en 2021): la variable y sus lags van a 1 y los deltas a 0.
    if not CAMBIOS["sin_enmascarar_internet"]:
        viejos = (train.foto_mes < months[-1]).to_numpy()
        train.loc[viejos, [c for c in internet if "_delta" not in c]] = RESTO_INTERNET
        train.loc[viejos, [c for c in internet if "_delta" in c]] = 0
    return train, elegidas or cols, np.asarray(y), w, objetivo


def puntaje(P: np.ndarray) -> np.ndarray:
    """De las tres probabilidades de la multiclase al puntaje que ordena: P(BAJA+2) + a · P(BAJA+1) (E019: a = 1)."""
    return P[:, 2] + CAMBIOS["puntaje_baja1"] * P[:, 1]


def base(train: pd.DataFrame, score: pd.DataFrame, seed: int, peso_fila=None, objetivo=None) -> np.ndarray:
    """El general (E019 ya no tiene especialista ni segunda etapa). Devuelve el puntaje de cada fila de score."""
    train, cols, y, w, objetivo = preparar(train, seed, peso_fila, objetivo)
    dtrain = lgb.Dataset(train[cols], label=y, weight=w, free_raw_data=True)
    if objetivo == "binaria":
        model = lgb.train({**PARAMS, "objective": "binary", "seed": seed}, dtrain, num_boost_round=NUM_ROUNDS)
        return model.predict(score[cols])
    model = lgb.train({**PARAMS, "objective": "multiclass", "num_class": 3, "seed": seed}, dtrain, num_boost_round=NUM_ROUNDS)
    return puntaje(model.predict(score[cols]))


# ---------------------------------------------------------------- v4: canaritos en el general

_GENERAL_ELEGIDAS = {}
# Las cachés de abajo (canaritos, bases fuera de muestra) son de todas las semillas: con las semillas en hilos, la primera
# que llega la calcula y las demás esperan. Sin esto las cinco la calculaban a la vez y LightGBM se cayó (segfault en
# V31). Reentrante: fuera_de_muestra llama a base, que puede llamar a general_canaritos.
_CANDADO_CACHE = threading.RLock()


def general_canaritos(train: pd.DataFrame, cols: list, canaritos: int = 50, repeticiones: int = 3) -> list:
    """Variables del general cuya ganancia media (importancia "gain") supera a la del mejor de 50 canaritos (columnas al
    azar). Una vez por ventana de entrenamiento, con una semilla fija: no cambia con la semilla del modelo."""
    clave = (tuple(sorted(train.foto_mes.unique())), len(train), int(train.numero_de_cliente.sum()), len(cols))
    with _CANDADO_CACHE:
        return _general_canaritos(train, cols, clave, canaritos, repeticiones)


def _general_canaritos(train, cols, clave, canaritos, repeticiones):
    if clave not in _GENERAL_ELEGIDAS:
        rng = np.random.default_rng(SEMILLA_OOF)
        t = train[(rng.random(len(train)) < UNDERSAMPLING) | (train.clase_ternaria != "CONTINUA").to_numpy()]
        y = t.clase_ternaria.map({"CONTINUA": 0, "BAJA+1": 1, "BAJA+2": 2}).to_numpy()
        X = t[cols].to_numpy(np.float32)
        ganancias = []
        for j in range(repeticiones):
            R = np.random.default_rng(SEMILLA_OOF + j).random((len(X), canaritos), dtype=np.float32)
            d = lgb.Dataset(np.hstack([X, R]), label=y, free_raw_data=True)
            m = lgb.train({**PARAMS, "objective": "multiclass", "num_class": 3, "seed": SEMILLA_OOF + j}, d,
                          num_boost_round=min(NUM_ROUNDS, 300))
            ganancias.append(m.feature_importance("gain"))
        g = np.mean(ganancias, axis=0)
        real, vara = g[:len(cols)], g[len(cols):].max()
        elegidas = [c for c, v in zip(cols, real) if v > vara]
        _GENERAL_ELEGIDAS[clave] = elegidas if len(elegidas) >= 20 else [cols[i] for i in np.argsort(-real)[:20]]
        print(f"  canaritos en el general: quedan {len(_GENERAL_ELEGIDAS[clave])} de {len(cols)} variables", flush=True)
    return _GENERAL_ELEGIDAS[clave]


# ---------------------------------------------------------------- v4: modelos del ensamble

def mezcla(*ps, pesos=None) -> np.ndarray:
    """Promedio de los puestos relativos (0 a 1) de cada modelo: escalas distintas, mismo peso salvo que se pida."""
    return np.average([pd.Series(p).rank(pct=True).to_numpy() for p in ps], axis=0, weights=pesos)


def xgb_general(train: pd.DataFrame, score: pd.DataFrame, seed: int) -> np.ndarray:
    """Un XGBoost con las mismas filas, clase, pesos y columnas que el general, y parámetros equivalentes."""
    import xgboost as xgb
    tr, cols, y, w, objetivo = preparar(train, seed)
    multi = objetivo == "multiclase"
    p = {"objective": "multi:softprob" if multi else "binary:logistic", "tree_method": "hist", "grow_policy": "lossguide",
         "max_leaves": PARAMS["num_leaves"], "max_depth": 0, "eta": PARAMS["learning_rate"],
         "colsample_bytree": PARAMS["feature_fraction"], "subsample": PARAMS.get("bagging_fraction", 1.0),
         "min_child_weight": 1, "seed": int(seed), "nthread": PARAMS.get("num_threads", 16), "verbosity": 0}
    if multi:
        p["num_class"] = 3
    m = xgb.train(p, xgb.QuantileDMatrix(tr[cols].to_numpy(np.float32), y, weight=w), NUM_ROUNDS)
    P = m.predict(xgb.DMatrix(score[cols].to_numpy(np.float32)))
    return puntaje(P) if multi else P


PARAMS_CATEDRA = dict(objective="binary", num_leaves=56, learning_rate=0.026, min_data_in_leaf=4997, feature_fraction=0.66,
                      bagging_fraction=0.77, bagging_freq=1, max_bin=255, verbosity=-1, num_threads=16)
RONDAS_CATEDRA = 400  # la media de las rondas del early stopping de FINAL0 en v3


def modelo_catedra(train: pd.DataFrame, score: pd.DataFrame, seed: int) -> np.ndarray:
    """Al estilo del final de v3 (B0 + H1b): crudas con lag y delta 1 y 2, BAJA+1 o BAJA+2 contra CONTINUA, sin
    submuestreo ni pesos, con los parámetros de B0. Las crudas son las de E019 (con rotas a NA y los percentiles)."""
    months = sorted(train.foto_mes.unique())[-TRAIN_MONTHS_BACK:]
    tr = train[train.foto_mes.isin(months)]
    crudas = [c for c in tr.columns if f"{c}_lag1" in tr.columns and c not in ("numero_de_cliente", "foto_mes")]
    cols = crudas + [f"{c}_{s}" for c in crudas for s in ("lag1", "lag2", "delta1", "delta2")]
    d = lgb.Dataset(tr[cols].to_numpy(np.float32), label=(tr.clase_ternaria != "CONTINUA").to_numpy().astype(int))
    m = lgb.train({**PARAMS_CATEDRA, "seed": int(seed), "num_threads": PARAMS["num_threads"]}, d, num_boost_round=RONDAS_CATEDRA)
    return m.predict(score[cols].to_numpy(np.float32))


def binaria_general(train: pd.DataFrame, score: pd.DataFrame, seed: int) -> np.ndarray:
    """El general como binaria (BAJA+1 o BAJA+2 contra CONTINUA), con todo lo demás igual (E027)."""
    return base(train, score, seed, objetivo="binaria")


ENSAMBLE = {"xgboost": xgb_general, "catedra": modelo_catedra, "binaria_ensamble": binaria_general}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def repartir(p_gen: np.ndarray, p_esp: np.ndarray) -> np.ndarray:
    """Probabilidad final de los clientes del segmento: CUÁNTO sube el grupo lo decide el máximo; A QUIÉN, los dos modelos.

    Si cada cliente del segmento se quedara con max(general, especialista), el grupo sube, que es lo que hace falta (el
    general lo subestima), pero a quién se sube lo elige el ruido del especialista: sube el cliente en el que el
    especialista, con sus ~6.000 filas, se pasó para arriba.
    Acá se separan las dos cosas:
      - el perfil: los valores de max(general, especialista) del segmento, ordenados. Son exactamente los mismos números
        que daría el máximo, así que por semilla entra a la lista la misma cantidad de deudores y con el mismo nivel;
      - el orden: esos valores se reparten de mayor a menor según el promedio de los logit de los dos modelos, con pesos
        iguales.
    """
    perfil = np.sort(np.maximum(p_gen, p_esp))[::-1]
    orden = np.argsort(-(logit(p_gen) + logit(p_esp)), kind="stable")
    out = np.empty_like(p_gen)
    out[orden] = perfil
    return out


# ---------------------------------------------------------------- segunda etapa sobre la franja dudosa

FRANJA = (3100, 11500)  # puestos de la base, sólo clientes del resto (internet == 1)
PARAMS_2 = dict(objective="binary", learning_rate=0.03, num_leaves=4, min_data_in_leaf=200, feature_fraction=0.3, lambda_l2=20.0,
                max_bin=63, verbosity=-1, num_threads=3)
RONDAS_2 = 100
MESES_DETRAS = 2             # meses de datos que tiene que tener detrás un mes para entrenar con él
SEMILLAS_FUERA = (800011,)  # semillas de la base fuera de muestra (se promedian si hay varias); se entrena una vez por proceso
VARIABLES_2 = [  # las que les ganan a los canaritos (percentil 95), en orden de ganancia
    "mcomisiones_mantenimiento_lag2", "mrentabilidad_max3", "mcuentas_saldo_delta2", "mcomisiones_mantenimiento_ratioavg3",
    "ccomisiones_mantenimiento_lag2", "ctrx_quarter_lag2", "mtarjeta_master_consumo_lag1", "ctrx_quarter_delta2", "cliente_edad_lag2",
    "chomebanking_transacciones_ratiomax3", "cproductos_ratiomax3", "ctrx_quarter_lag1", "mcuenta_corriente", "cliente_edad_lag1",
    "mtransferencias_recibidas_ratioavg3", "mtarjeta_master_consumo_ratiolag1", "ctransferencias_recibidas_ratioavg3",
    "chomebanking_transacciones_ratioavg3", "ccaja_ahorro_lag2", "cproductos_ratioavg3", "mcomisiones_mantenimiento",
    "ccomisiones_mantenimiento_ratioavg3", "mpagomiscuentas_delta2", "ctarjeta_visa_transacciones_ratioavg3",
    "ctarjeta_visa_transacciones_ratiomax3",
]
_FUERA = {}  # train -> {mes: base fuera de muestra}; se calcula una vez por proceso


def puesto(p):
    """Puesto de cada cliente (0 = el más probable), con orden estable."""
    r = np.empty(len(p), int)
    r[np.argsort(-p, kind="stable")] = np.arange(len(p))
    return r


def grupo(internet, p, franja=None):
    """Clientes de un mes en la franja de puestos de p y del resto (internet == 1)."""
    franja = franja or FRANJA
    r = puesto(p)
    return (r >= franja[0]) & (r < franja[1]) & (np.asarray(internet) == RESTO_INTERNET)


def calibracion(l, y):
    """Regresión logística y ~ c0 + c1 * l, por Newton."""
    A = np.column_stack([np.ones(len(l)), l])
    c = np.zeros(2)
    for _ in range(40):
        mu = 1 / (1 + np.exp(-A @ c))
        c += np.linalg.solve(A.T @ (A * (mu * (1 - mu))[:, None]) + 1e-8 * np.eye(2), A.T @ (y - mu))
    return c


def meses_de_entrenamiento(train):
    """Los dos últimos meses con clase que tienen MESES_DETRAS meses de datos antes (mayo y junio en la entrega)."""
    meses = sorted(int(m) for m in train.foto_mes.unique())
    return [m for i, m in enumerate(meses) if i >= MESES_DETRAS][-2:]


def base_fuera_de_muestra(train: pd.DataFrame) -> dict:
    """{mes: (clientes, probabilidad de la base para esas filas, sin haber visto su clase)} para mayo y junio.

    UN corte: la base se entrena con las filas con clase ANTERIORES al primero de los dos meses (clase <= 202104) y puntúa
    los dos; si SEMILLAS_FUERA trae más de una, se promedian. Se calcula una vez por proceso y vale para todas las semillas de la entrega.
    """
    meses = meses_de_entrenamiento(train)
    clave = (tuple(meses), len(train), int(train.numero_de_cliente.sum()))
    with _CANDADO_CACHE:
        return _base_fuera_de_muestra(train, meses, clave)


def _base_fuera_de_muestra(train, meses, clave):
    if clave not in _FUERA:
        out = {}
        if meses:
            antes = train[(train.foto_mes < meses[0]).to_numpy()]
            a_puntuar = train[train.foto_mes.isin(meses).to_numpy()].drop(columns=["clase_ternaria"])
            p = np.mean([np.asarray(base(antes, a_puntuar, s), dtype=np.float64) for s in SEMILLAS_FUERA], axis=0)
            mes = a_puntuar.foto_mes.to_numpy()
            ids = a_puntuar.numero_de_cliente.to_numpy()
            out = {m: (ids[mes == m], p[mes == m]) for m in meses}
        _FUERA[clave] = out
    return _FUERA[clave]


def segunda_etapa(train: pd.DataFrame, score: pd.DataFrame, seed: int, p_base: np.ndarray) -> np.ndarray:
    """Probabilidad corregida, del mismo largo que p_base (la de base() para las filas de score, con esa semilla).

    Entrena con las franjas de mayo y junio (por el orden de la base fuera de muestra), sin marca internet y sin las BAJA+1;
    clase: BAJA+2 contra CONTINUA; punto de partida c0 + c1 * logit(base), la calibración de la base dentro de la franja.
    Sin meses con qué entrenar (clase sólo hasta abril) devuelve p_base sin tocar.
    """
    p = np.asarray(p_base, dtype=np.float64).copy()
    fuera = base_fuera_de_muestra(train)
    if not fuera:
        return p
    X, y, l = [], [], []
    for m, (ids_m, pm) in fuera.items():
        t = train[(train.foto_mes == m).to_numpy()]
        assert np.array_equal(t.numero_de_cliente.to_numpy(), ids_m), "la base fuera de muestra no está alineada con las filas del mes"
        clase = t.clase_ternaria.to_numpy()
        k = grupo(t.internet.to_numpy(), pm) & (clase != "BAJA+1")
        X.append(t.loc[k, VARIABLES_2]); y.append((clase[k] == "BAJA+2").astype(float)); l.append(logit(pm[k]))
    X, y, l = pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(l)
    if len(y) < 2 * PARAMS_2["min_data_in_leaf"] or y.min() == y.max():  # con muy pocas filas o una sola clase no hay con qué aprender
        return p
    c = calibracion(l, y)
    modelo = lgb.train({**PARAMS_2, "seed": seed}, lgb.Dataset(X, label=y, init_score=c[0] + c[1] * l), num_boost_round=RONDAS_2)
    mes, internet = score.foto_mes.to_numpy(), score.internet.to_numpy()
    for m in np.unique(mes):  # mes por mes: el puesto y el reparto son de los clientes de UN mes
        f = np.flatnonzero(mes == m)
        idx = f[grupo(internet[f], p[f])]
        if not len(idx):
            continue
        s = modelo.predict(score.iloc[idx][VARIABLES_2], raw_score=True) + c[0] + c[1] * logit(p[idx])
        s = pd.Series(s).rank().to_numpy() + pd.Series(p[idx]).rank().to_numpy()  # suma de puestos: corregido + base
        orden = np.argsort(-s, kind="stable")
        nuevo = np.empty(len(idx))
        nuevo[orden] = np.sort(p[idx])[::-1]  # los mismos valores, repartidos por el orden nuevo
        p[idx] = nuevo
    return p




# ---------------------------------------------------------------- franja dudosa: diez enfoques (CAMBIOS["franja"] = 1 a 10)
#
# La franja son los clientes que el general deja entre 0,6 y 3 veces la cantidad de envíos: ahí se decide la lista.
# Todos los enfoques aprenden de la franja de los meses de entrenamiento puntuados FUERA DE MUESTRA (cross-fitting por
# mes: cada mes de la ventana lo puntúa el general entrenado con los otros meses) y, salvo el 5 y el 9, reordenan la
# franja del mes que se puntúa sin cambiar los valores de probabilidad (como la segunda etapa de la primera competencia).
FRANJA_REL = (0.6, 3.0)
SEMILLA_OOF = 800011
TENDENCIA = ("_delta1", "_delta2", "_ratiolag1", "_ratioavg3", "_ratiomax3", "_tend3", "_min3", "_max3", "_avg3")
_OOF = {}
# Aparte de _CANDADO_CACHE: los meses se entrenan en hilos que pueden necesitar aquél (canaritos del general).
_CANDADO_OOF = threading.RLock()


def _clave(train):
    return (tuple(sorted(train.foto_mes.unique())), len(train), int(train.numero_de_cliente.sum()))


def fuera_de_muestra(train: pd.DataFrame, objetivo="multiclase") -> pd.Series:
    """Probabilidad de cada fila de entrenamiento según el general entrenado con los otros meses de la ventana."""
    clave = _clave(train) + (objetivo,)
    with _CANDADO_OOF:
        return _fuera_de_muestra(train, objetivo, clave)


def _fuera_de_muestra(train, objetivo, clave):
    if clave not in _OOF:
        def uno(m):
            f = (train.foto_mes == m).to_numpy()
            return f, base(train[~f], train[f].drop(columns=["clase_ternaria"]), SEMILLA_OOF, objetivo=objetivo)

        meses = sorted(train.foto_mes.unique())
        p = pd.Series(np.nan, index=train.index)
        # En paralelo: la calcula la primera semilla que llega y las demás esperan sin hacer nada.
        with ThreadPoolExecutor(max_workers=len(meses)) as hilos:
            for f, v in hilos.map(uno, meses):
                p[f] = v
        _OOF[clave] = p
    return _OOF[clave]


def en_franja(p, n_mes):
    k = CORTE["f"] * n_mes
    r = puesto(np.asarray(p))
    return (r >= FRANJA_REL[0] * k) & (r < FRANJA_REL[1] * k)


def franja_train(train: pd.DataFrame, poof: pd.Series, sin_baja1=True):
    """Filas de la franja de cada mes de entrenamiento (según el puntaje fuera de muestra) y su clase BAJA+2."""
    k = np.zeros(len(train), bool)
    for m in train.foto_mes.unique():
        f = (train.foto_mes == m).to_numpy()
        k[f] = en_franja(poof[f].to_numpy(), f.sum())
    if sin_baja1:
        k &= (train.clase_ternaria != "BAJA+1").to_numpy()
    t = train[k]
    return t, (t.clase_ternaria == "BAJA+2").to_numpy().astype(float), poof[k].to_numpy()


PESO_BASE_FRANJA = 1.0  # peso del puesto de p en la suma de puestos (lo mueve la búsqueda de la franja)
FRANJA_BAJA1 = "fuera"  # las BAJA+1 de la franja de entrenamiento: "fuera", "negativa" o "positiva" (enfoques 1 y 2)
FRANJA_INIT = True      # la segunda etapa parte del puntaje fuera de muestra del general (enfoques 1 y 2)


def reordenar(p, idx, s_nuevo):
    """Los mismos valores de p en la franja, repartidos por la suma de puestos del puntaje nuevo y de p."""
    s = pd.Series(np.asarray(s_nuevo)).rank().to_numpy() + PESO_BASE_FRANJA * pd.Series(p[idx]).rank().to_numpy()
    out = p.copy()
    nuevo = np.empty(len(idx))
    nuevo[np.argsort(-s, kind="stable")] = np.sort(p[idx])[::-1]
    out[idx] = nuevo
    return out


def _cols(train):
    return [c for c in train.columns if c not in ("clase_ternaria", "numero_de_cliente", "foto_mes") and not historia_de_resumen(c)
            and not c.endswith(CAMBIOS["sacar"])]


def _lgb(X, y, params, rondas, seed, init=None, grupo=None):
    d = lgb.Dataset(X, label=y, init_score=init, group=grupo)
    return lgb.train({**params, "seed": seed, "verbosity": -1, "num_threads": PARAMS["num_threads"]}, d, num_boost_round=rondas)


def puntaje_franja(enfoque, train, score, seed, p, idx, intermedio=None):
    """El puntaje nuevo de las filas idx del mes que se puntúa, según el enfoque."""
    cols = _cols(train)
    poof = fuera_de_muestra(train)
    if enfoque in (1, 2):  # segunda etapa con cross-fitting; 2: variables elegidas con canaritos
        t, y, l = franja_train(train, poof, sin_baja1=FRANJA_BAJA1 == "fuera")
        if FRANJA_BAJA1 == "positiva":
            y = np.maximum(y, (t.clase_ternaria == "BAJA+1").to_numpy())
        if enfoque == 2:
            cols = le_ganan_al_ruido(t, cols, y.astype(int))
        if not FRANJA_INIT:
            return _lgb(t[cols], y, PARAMS_2, RONDAS_2, seed).predict(score.iloc[idx][cols], raw_score=True)
        c = calibracion(logit(l), y)
        m = _lgb(t[cols], y, PARAMS_2, RONDAS_2, seed, init=c[0] + c[1] * logit(l))
        return m.predict(score.iloc[idx][cols], raw_score=True) + c[0] + c[1] * logit(p[idx])
    if enfoque == 3:  # cascada: el 15% de arriba, con un modelo más grande
        k = np.zeros(len(train), bool)
        for mm in train.foto_mes.unique():
            f = (train.foto_mes == mm).to_numpy()
            k[f] = puesto(poof[f].to_numpy()) < 0.15 * f.sum()
        k &= (train.clase_ternaria != "BAJA+1").to_numpy()
        t = train[k]
        m = _lgb(t[cols], (t.clase_ternaria == "BAJA+2").astype(float), dict(objective="binary", learning_rate=0.03, num_leaves=63,
                 min_data_in_leaf=100, feature_fraction=0.5, max_bin=255), 300, seed)
        return m.predict(score.iloc[idx][cols])
    if enfoque == 4:  # lambdarank: optimiza el orden dentro de la franja
        # LightGBM acepta hasta 10.000 filas por consulta y la franja de un mes tiene unas 10.000: cada mes se parte en
        # consultas de hasta 5.000 filas al azar (E037 se cayó por esto).
        t, y, _ = franja_train(train, poof)
        orden = np.random.default_rng(seed).permutation(len(t))
        t = t.iloc[orden]
        t = t.assign(_q=t.groupby("foto_mes").cumcount() // 5000).sort_values(["foto_mes", "_q"], kind="stable")
        y = (t.clase_ternaria == "BAJA+2").astype(int).to_numpy()
        grupos = t.groupby(["foto_mes", "_q"], sort=True).size().to_numpy()
        m = _lgb(t[cols], y, dict(objective="lambdarank", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
                 feature_fraction=0.5, max_bin=255, lambdarank_truncation_level=5000), 200, seed, grupo=grupos)
        return m.predict(score.iloc[idx][cols])
    if enfoque in (6, 10):  # sólo variables de tendencia: 6 con un árbol, 10 con una logística
        tc = [c for c in cols if c.endswith(TENDENCIA)]
        t, y, _ = franja_train(train, poof)
        if enfoque == 6:
            m = _lgb(t[tc], y, dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
                     feature_fraction=0.5, max_bin=255), 200, seed)
            return m.predict(score.iloc[idx][tc])
        from sklearn.linear_model import LogisticRegression
        X = t[tc].to_numpy(np.float64)
        med, esc = np.nanmedian(X, axis=0), np.nanpercentile(X, 90, axis=0) - np.nanpercentile(X, 10, axis=0)
        esc = np.where(esc > 0, esc, 1.0)
        norm = lambda Z: np.clip(np.nan_to_num((Z - med) / esc), -5, 5)
        lr = LogisticRegression(C=0.05, max_iter=500).fit(norm(X), y)
        return lr.decision_function(norm(score.iloc[idx][tc].to_numpy(np.float64)))
    if enfoque == 7:  # baja inminente: un modelo de BAJA+1 con las BAJA+1 ya conocidas de m - 1
        partes = [train.assign(_y=(train.clase_ternaria == "BAJA+1").astype(float))]
        if intermedio is not None and len(intermedio):
            partes.append(intermedio.assign(_y=intermedio.baja1.astype(float)))
        t = pd.concat([x[cols + ["_y"]] for x in partes], ignore_index=True)
        keep = (t._y == 1).to_numpy() | (np.random.default_rng(seed).random(len(t)) < UNDERSAMPLING)
        t = t[keep]
        m = _lgb(t[cols], t._y, dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                 feature_fraction=0.5, max_bin=255), 300, seed)
        return m.predict(score.iloc[idx][cols])
    if enfoque == 8:  # meta-modelo: la multiclase y la binaria fuera de muestra más algunas variables
        pbin = fuera_de_muestra(train, "binaria")
        t, y, l = franja_train(train, poof)
        extra = ["internet", "ctrx_quarter", "mcuentas_saldo", "cliente_antiguedad", "cproductos", "mpayroll"]
        X = np.column_stack([logit(l), logit(pbin[t.index].to_numpy()), t[extra].to_numpy(np.float64)])
        m = _lgb(X, y, dict(PARAMS_2), RONDAS_2, seed)
        pb = base(train, score.iloc[idx], seed, objetivo="binaria")
        Xs = np.column_stack([logit(p[idx]), logit(pb), score.iloc[idx][extra].to_numpy(np.float64)])
        return m.predict(Xs)
    if enfoque == 11:  # TabPFN (v3 H30): aprende la franja con VARIABLES_2 y el logit de la base, en un proceso aparte
        t, y, l = franja_train(train, poof)
        return tabpfn_puntaje(t[VARIABLES_2].to_numpy(np.float32), logit(l), y.astype(int),
                              score.iloc[idx][VARIABLES_2].to_numpy(np.float32), logit(p[idx]), seed)
    raise ValueError(enfoque)


# TabPFN usa torch, que choca con LightGBM en el mismo proceso en macOS (cada uno trae su OpenMP): corre aparte y los
# datos van y vuelven por archivos .npy.
TABPFN_HIJO = r"""
import sys
import numpy as np
import torch
from tabpfn import TabPFNClassifier
d, seed = sys.argv[1], int(sys.argv[2])
m = TabPFNClassifier(device="mps" if torch.backends.mps.is_available() else "cpu", random_state=seed)
m.fit(np.load(f"{d}/X.npy"), np.load(f"{d}/y.npy"))
np.save(f"{d}/p.npy", m.predict_proba(np.load(f"{d}/Z.npy"))[:, 1])
"""


def tabpfn_puntaje(M, l, y, Z, lz, seed, max_filas=10_000):
    """Entrena TabPFN con hasta 10.000 filas de la franja (todas las BAJA+2 y CONTINUA al azar) y puntúa Z."""
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    import config
    rng = np.random.default_rng(int(seed))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    filas = np.concatenate([pos, rng.choice(neg, min(len(neg), max_filas - len(pos)), replace=False)])
    (config.CACHE / "v4").mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="tabpfn_", dir=config.CACHE / "v4"))
    try:
        np.save(d / "X.npy", np.column_stack([M[filas], l[filas]]).astype(np.float32))
        np.save(d / "y.npy", y[filas])
        np.save(d / "Z.npy", np.column_stack([Z, lz]).astype(np.float32))
        subprocess.run([sys.executable, "-c", TABPFN_HIJO, str(d), str(int(seed))], check=True)
        return np.load(d / "p.npy")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def corte(prob, train, score):
    """Enfoque 9: el 60% de arriba de los envíos va siempre; en la franja, quien tenga P(BAJA+2) calibrada (isotónica sobre
    la franja fuera de muestra) mayor que 2,5%. Los demás enfoques usan el 3,3% fijo."""
    n = int(round(CORTE["f"] * len(prob)))
    if CAMBIOS["franja"] != 9:
        return n
    from sklearn.isotonic import IsotonicRegression
    t, y, l = franja_train(train, fuera_de_muestra(train), sin_baja1=False)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(l, y)
    r = puesto(np.asarray(prob))
    seguros = int((r < FRANJA_REL[0] * n).sum())
    f = en_franja(prob, len(prob))
    return seguros + int((iso.predict(np.asarray(prob)[f]) >= 27_500 / 1_100_000).sum())


# ---------------------------------------------------------------- el modelo

def peso_franja(train: pd.DataFrame) -> pd.Series:
    """Enfoque 5: el doble de peso a las filas de entrenamiento que caen en la franja (fuera de muestra)."""
    poof = fuera_de_muestra(train)
    k = np.zeros(len(train), bool)
    for m in train.foto_mes.unique():
        f = (train.foto_mes == m).to_numpy()
        k[f] = en_franja(poof[f].to_numpy(), f.sum())
    return pd.Series(np.where(k, 2.0, 1.0), index=train.index)


def train_and_predict(train: pd.DataFrame, score: pd.DataFrame, seed: int, intermedio=None) -> np.ndarray:
    enfoque = CAMBIOS["franja"]
    p = base(train, score, seed, peso_fila=peso_franja(train) if enfoque == 5 else None)
    otros = [f(train, score, seed) for k, f in ENSAMBLE.items() if CAMBIOS[k]]
    otros += [miembro(c, train, score, seed) for c in CAMBIOS["miembros"]]
    if otros:
        p = mezcla(p, *otros, pesos=[1.0] + [CAMBIOS["peso_miembros"]] * len(otros))
    if enfoque in (0, 5, 9):
        return p
    idx = np.flatnonzero(en_franja(p, len(p)))
    return reordenar(p, idx, puntaje_franja(enfoque, train, score, seed, p, idx, intermedio))


# ---------------------------------------------------------------- v4: hiperparámetros de una búsqueda

def aplicar(params: dict) -> None:
    """Pone los hiperparámetros de una búsqueda: los de LightGBM van a PARAMS; num_rounds, submuestreo, peso_baja1 y
    peso_baja2 a sus constantes."""
    global NUM_ROUNDS, UNDERSAMPLING, PESO_BAJA1, PESO_BAJA2
    p = dict(params)
    NUM_ROUNDS = int(p.pop("num_rounds", NUM_ROUNDS))
    UNDERSAMPLING = float(p.pop("submuestreo", UNDERSAMPLING))
    PESO_BAJA1 = float(p.pop("peso_baja1", PESO_BAJA1))
    PESO_BAJA2 = float(p.pop("peso_baja2", PESO_BAJA2))
    PARAMS.update(p)


def aplicar_franja(fp: dict) -> None:
    """Pone los parámetros de la franja de una búsqueda (v4/hpo_franja.py): franja_desde y franja_hasta van a
    FRANJA_REL, rondas a RONDAS_2, peso_base a PESO_BASE_FRANJA, baja1 a FRANJA_BAJA1, init a FRANJA_INIT y el resto a
    PARAMS_2."""
    global FRANJA_REL, RONDAS_2, PESO_BASE_FRANJA, FRANJA_BAJA1, FRANJA_INIT
    p = dict(fp)
    FRANJA_BAJA1 = str(p.pop("baja1", FRANJA_BAJA1))
    FRANJA_INIT = bool(p.pop("init", FRANJA_INIT))
    FRANJA_REL = (float(p.pop("franja_desde", FRANJA_REL[0])), float(p.pop("franja_hasta", FRANJA_REL[1])))
    RONDAS_2 = int(p.pop("rondas", RONDAS_2))
    PESO_BASE_FRANJA = float(p.pop("peso_base", PESO_BASE_FRANJA))
    PARAMS_2.update(p)


_COPIAS = {}
_CANDADO = threading.Lock()


def miembro(cambios: dict, train: pd.DataFrame, score: pd.DataFrame, seed: int) -> np.ndarray:
    """El general con otras constantes sobre las de E019 ("meses" va a TRAIN_MONTHS_BACK, lo demás a aplicar()). Corre en una copia
    aparte de este módulo: cambiar las constantes de éste pisaría las del general, que puede estar entrenando otra
    semilla en otro hilo."""
    clave = repr(sorted(cambios.items()))
    with _CANDADO:
        if clave not in _COPIAS:
            spec = importlib.util.spec_from_file_location(f"{__name__}_miembro{len(_COPIAS)}", __file__)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            # Parte de E019 (no de params= del general, que pisaría las claves que el miembro no fija); los hilos
            # son los del general, que pone el arnés.
            m.PARAMS = {**m._DEL_ARCHIVO[0], "num_threads": PARAMS["num_threads"]}
            m.NUM_ROUNDS, m.UNDERSAMPLING, m.PESO_BAJA1, m.PESO_BAJA2 = m._DEL_ARCHIVO[1:]
            c = dict(cambios)
            m.TRAIN_MONTHS_BACK = int(c.pop("meses", TRAIN_MONTHS_BACK))
            m.aplicar(c)
            _COPIAS[clave] = m
    return _COPIAS[clave].base(train, score, seed)


_DEL_ARCHIVO = (dict(PARAMS), NUM_ROUNDS, UNDERSAMPLING, PESO_BAJA1, PESO_BAJA2)  # E019, antes de params=
if CAMBIOS["params"]:
    aplicar(CAMBIOS["params"])
if CAMBIOS["franja_params"]:
    aplicar_franja(CAMBIOS["franja_params"])
