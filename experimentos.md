# Plan de experimentos

Objetivo: el mejor modelo para marcar qué clientes de **202108** son BAJA+2, con 36 meses de historia.
Ganancia: +1.072.500 por cada BAJA+2 estimulado, −27.500 por cada estímulo.

Cada ítem es UNA idea, que se corre como un experimento con `run.py` (salvo la fase 0, que son herramientas). Los ids
(P1, B2, E6…) son los que se pasan en `--idea` y los que muestra la página. La regla de adopción y el protocolo están
en `program.md`; acá va qué probar y por qué. Las fuentes están al final.

## Fase 0: mirar los datos (una vez, antes del loop)

- **P1. Variables rotas.** `python tools/fase0.py` → `fase0/rotas.tsv`: variables que en algún mes están todas en cero,
  vacías o constantes. Las que aparezcan se pasan a NA en ese mes dentro de `make_features` (experimento propio).
- **P2. Deriva.** `fase0/deriva.tsv`: montos que se corren más de 25% contra la mediana de los 12 meses anteriores
  (inflación, aguinaldo en junio y diciembre). Alimenta E6.
- **P3. Fuga.** `python tools/fuga.py` cada vez que cambie `make_features`: las variables de un mes no pueden cambiar
  al agregar meses futuros. Tampoco se usan variables que identifiquen el mes (`foto_mes`, mes calendario).
  Fuente: *The Hidden Costs of 99% Accuracy*.
- **P4. BAJA+2 por mes.** `fase0/clase_por_mes.tsv`: tasa por mes y razón contra dos meses antes, en todos los años.
  Alimenta E7.

## Básicos

- **B0. Base.** El modelo de clase del curso tal como está en `train.py`: variables crudas, un mes de entrenamiento,
  binaria BAJA+2. Es el primer experimento y el primer vigente.
- **B1. Clase objetivo.** BAJA+2 sola / BAJA+1 ∪ BAJA+2 como positivos / multiclase rankeando por
  P(BAJA+2) + α·P(BAJA+1) / BAJA+1 como positiva con peso 0,25 y 0,5. La ganancia sólo paga BAJA+2: lo que se mide
  es si sumar BAJA+1 al entrenamiento le da al modelo más señal de "cliente que se va".
- **B2. Lags y deltas.** Lag y delta (x − lag) de 1 y 2 meses de todas las variables. Requiere `HISTORIA_MESES = 2`.
- **B3. Submuestreo de CONTINUA.** 1,0 / 0,5 / 0,2 / 0,1 con `prepare.muestra_continua` (pareado por semilla).
  Además de la ganancia, mirar el tiempo: con más meses de entrenamiento (E1) va a hacer falta.

## Prioridad 1: historia y ventanas

Fuentes: *A Rolling-Window Framework for Churn Prediction*; *How Early Is Early Enough?*

- **E1. Meses de entrenamiento.** 1 / 3 / 6 / 12 / 24 / todos (`MESES_ENTRENAMIENTO`). Escalar `min_data_in_leaf` con
  la cantidad de filas y usar el submuestreo de B3 para no pasarse del presupuesto.
- **E2. Sin meses atípicos.** Con 12 meses o más, sacar del entrenamiento los meses de pandemia si están en el rango
  (abril a diciembre de 2020).
- **E3. Peso por antigüedad del mes.** Peso de cada fila = 0,5^(edad del mes / h), con h ∈ {3, 6, 12}.
- **E4. Largo de la ventana de features.** Lags 1..k con k ∈ {2, 3, 6, 12}; promedio, mínimo, máximo y tendencia
  sobre 3, 6 y 12 meses. Reportar la curva entera (ganancia contra k), no sólo el mejor punto: el paper muestra que
  el codo depende del diseño. `prepare` ya garantiza que un mes entra al entrenamiento sólo si tiene detrás toda la
  historia que declara `HISTORIA_MESES`, así que una variable "contra el máximo de la ventana" vale lo mismo en el
  entrenamiento y en el mes a puntuar.
- **E5. Memoria larga.** Caída contra el máximo de 12 meses (saldo, transacciones, consumo) y meses desde el último
  sueldo cobrado.

## Prioridad 1: estacionalidad

Fuente: *Seasonal false alarms in customer churn and decline early-warning systems*.

- **E6. Deltas interanuales.** x − lag12 y x / lag12 para montos y conteos, en lugar de delta1 y además de delta1.
  Hipótesis: el delta contra el mes anterior mezcla aguinaldo e inflación con una caída real. Variantes: interanual
  sobre el percentil del mes (rank dentro de `foto_mes`, con el cero fijo) y percentil sin interanual.
- **E7. Corte con estacionalidad.** Estimar cuántos BAJA+2 va a tener el mes a puntuar con la razón contra dos meses
  antes de los mismos meses de años anteriores (P4), y fijar la cantidad de envíos con eso (`CORTE` en `train.py`).
  Validarlo prediciendo el control (202008) desde 202006.

## Prioridad 2: hiperparámetros

Fuentes: *Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch*;
*Agentic Search Spaces for Tabular Machine Learning*; *When Is an LLM Worth It for Hyperparameter Optimization?*

Regla general: el agente no mueve hiperparámetros a mano. Diseña el espacio, Optuna busca (`tools/hpo.py`), y el
ganador se copia a `train.py` y se corre como experimento. Objetivo de la búsqueda: el ensamble, media de los folds
de selección, con las semillas del bloque de búsqueda.

- **E8. Espacio ampliado.** El agente escribe `hpo/espacio_ampliado.json` con una línea de justificación por
  parámetro en `hpo/espacio_ampliado.md`. Además de los 4 del espacio base (learning_rate, feature_fraction,
  num_leaves, min_data_in_leaf), considerar: num_rounds, bagging_fraction + bagging_freq, pos/neg_bagging_fraction,
  lambda_l1, lambda_l2, min_gain_to_split, min_sum_hessian_in_leaf, max_depth, max_bin (15 a 1023), extra_trees,
  path_smooth y submuestreo; con rangos que escalen con las filas de entrenamiento. Comparar contra el espacio base con
  la misma cantidad de trials. Si el ganador queda en el borde de un rango (`estado.md` lo marca), ampliarlo y repetir.
- **E9. Arranque.** Primer trial = la configuración vigente (`--arranque`). Comparar TPE con y sin arranque al mismo
  presupuesto.
- **E10. Híbrido agente + TPE.** Cada 10 trials el agente lee `hpo/<nombre>/estado.md` (mejores trials,
  importancias, parámetros en el borde) y propone hasta 2 trials con `tools/hpo.py proponer`; el resto los elige TPE.
  Comparar contra TPE puro al mismo presupuesto.

## Prioridad 3: variables generadas por agentes

Fuente: *Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation (MALMAS)*.

- **E11. Generación por roles con memoria.** Cuatro roles: temporal, interanual, agregados por grupo, interacciones.
  Por ronda se eligen 2 roles; 4 rondas como máximo. Una variable nueva entra sólo si supera al mejor de 20 canaritos
  (columnas al azar) en importancia y suma en el ensamble. Cada intento va a `features_probadas.tsv`
  (variable, rol, resultado) para no repetirlo, y al cerrar la ronda se agregan a `LECCIONES.md` tres líneas de por qué
  funcionó o no. Expectativa baja (en churn de telecomunicaciones el paper ganó 0,006 de AUC): priorizar el rol temporal.
- **E12. Agregados por grupo sin fuga.** Tasa histórica de BAJA+2 del grupo del cliente (por ejemplo, bandas de
  transacciones × antigüedad), calculada sólo con meses con clase hasta m − 2 y pasada como variable.

## Prioridad 3: corte y semillas

Fuentes: *Calibrated, cost-sensitive, explainable ensemble learning for customer churn prediction in retail
banking*; *The Hidden Costs of 99% Accuracy*.

- **E13. Corte por probabilidad calibrada.** Calibrar P(BAJA+2) con regresión isotónica en un mes fuera de muestra (el
  submuestreo y los pesos la descalibran) y estimular a quien tenga P > 27.500 / 1.072.500 ≈ 2,56%
  (`CORTE = {"tipo": "umbral", ...}`). Comparar contra el corte por cantidad (E7) por la ganancia que se pierde en cada
  fold con cada regla.
- **E14. Tamaño del semillerío.** Curva de la ganancia del ensamble con 5, 10, 20, 40 y 60 semillas (las primeras N
  del bloque final). N se fija por la curva; nunca se eligen subconjuntos de semillas mirando una métrica.

## Prioridad 4: proceso del loop

Fuentes: *Can We Predict Before Executing Machine Learning Agents? (FOREAGENT)*; *Recovering Wasted Compute in
Autoresearch Agents*.

- **E15. Ordenar la cola por predicción.** Antes de cada tanda, el agente ordena las ideas pendientes por ganancia
  esperada (con `LECCIONES.md` y la fase 0) y anota la predicción en la hipótesis del experimento. Sirve para ordenar,
  no para descartar: en el paper la predicción acierta 61,5% de las comparaciones de a pares.
- **E16. Volver atrás.** Si 5 experimentos seguidos se descartan, retomar la segunda mejor rama en vez de seguir
  siempre desde el último vigente.

## Decisión final (escrita antes de mirar 202108)

1. Finalistas: hasta 3, los mejores confirmados por la media de los folds de selección.
2. Se elige entre ellos con el fold de reserva (202103), una sola lectura: `python tools/sellado.py reserva ...`.
3. Semillas: las primeras N del bloque final (N de E14). Corte: la regla de E7 o de E13, la que pierda menos en los
   folds.
4. Recién entonces se mide 202108 con `tools/sellado.py test`, con el corte ya fijado. Se reportan todos los
   finalistas medidos y el modelo de referencia, si lo hay.

## No hacer

- Redes profundas o modelos de series de tiempo: en churn tabular el GBDT les gana (*ChurnNet*).
- Foundation models tabulares (*TabFM-Auto*): con este volumen no entran en el presupuesto por corrida.
- Variables que identifiquen el mes.
- Mover hiperparámetros a mano dentro del loop.
- Mirar 202108 dentro del loop.
- Elegir subconjuntos de semillas o el corte mirando una métrica.

## Fuentes

- Seasonal false alarms in customer churn and decline early-warning systems — https://arxiv.org/abs/2608.18174
- How Early Is Early Enough? Design-Dependent Observation-Window Sufficiency in Subscription Churn Prediction — https://arxiv.org/abs/2607.00473
- A Rolling-Window Framework for Churn Prediction and Behavioral Driver Identification — https://arxiv.org/abs/2606.06776
- Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch — https://arxiv.org/abs/2603.24647
- Agentic Search Spaces for Tabular Machine Learning — https://arxiv.org/abs/2609.16309
- When Is an LLM Worth It for Hyperparameter Optimization? — https://arxiv.org/abs/2606.21641
- Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data — https://arxiv.org/abs/2604.20261
- Calibrated, cost-sensitive, explainable ensemble learning for customer churn prediction in retail banking — https://doi.org/10.3389/frai.2026.1889639
- The Hidden Costs of 99% Accuracy: A Trustworthiness Audit of the Telco Customer Churn Benchmark — https://arxiv.org/abs/2610.00118
- Can We Predict Before Executing Machine Learning Agents? — https://arxiv.org/abs/2601.05930
- Recovering Wasted Compute in Autoresearch Agents — https://arxiv.org/abs/2608.10424
- ChurnNet: A Optimized Modern AI for Churn Prediction — https://arxiv.org/abs/2606.00169
- TabFM-Auto: Self-Evolving Pipelines for Tabular Foundation Models — https://arxiv.org/abs/2609.37989
