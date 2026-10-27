# Plan de experimentos

**Regla de la Segunda Competencia:** el modelo se define **sólo con datos de 2019**. El presente es 201912: se ven los
datos hasta 201912 y la clase como se la sabía entonces (completa hasta 201910, 201911 sólo con BAJA+1, 201912 sin
clase). Después, en 2020, el modelo sólo se reentrena todos los meses con la ventana de 6 meses, sin cambiarlo. En
202007, 202008 y 202009 (el aula) se simula estar en ese momento y se implementan los cambios que hagan falta con lo que
llega.

**Cómo se mide cada experimento (igual para todos):** el test son **201908, 201909 y 201910**, los tres últimos meses
con clase completa en 201912. Para cada uno se entrena con lo que se sabía en ese mes (la misma regla que en
producción: clase completa de m − 5 a m − 2 y BAJA+1 de m − 1), se arma la lista con el corte del modelo y se la mide con
la clase. Decide la **ganancia de las tres listas** (+1.072.500 por BAJA+2 estimulado, −27.500 por estímulo); se
informan también la meseta de la curva y la AUC. Los que se confirman se corren además con **data modificada simulada
sobre esos mismos meses** (montos inflados de golpe, variables rotas, columnas que faltan), para preparar el aula sin
mirar 2020.

**Regla de adopción:** screening con 5 semillas: pasa a confirmación si la ganancia media por mes mejora más de 3 M
contra el vigente y mejora en al menos 2 de los 3 meses. Confirmación con 20 semillas frescas: la lista del ensamble gana
más y la diferencia pareada por semilla supera 2 errores estándar.

**2020 no decide nada.** Al final, el modelo elegido se reentrena mes a mes en 2020 (`tools/produccion.py`) y se
informa qué habría ganado. Ningún experimento se elige mirando ese número.

Cada ítem es una idea; las variantes se corren como experimentos separados. Los ids son los que se pasan en `--idea`.

## Base

- **B0. Modelo de la primera competencia.** "Tips Ale Bolaño" tal como se entregó (general multiclase + especialista
  del grupo riesgoso + segunda etapa sobre la franja dudosa), llevado a estos datos con lo que se ve en 2019: tabla de
  variables rotas de 2019, el grupo riesgoso es `internet == 0` (en 2019 la variable está al revés que en 2021), franja
  y corte escalados al tamaño de estas listas (3,3% del mes). Fuente: github.com/emavaldez/primeracompetencia.

## Fase 0: mirar los datos

- **P1. Variables rotas detectadas en cada mes.** La base usa una tabla fija de 2019; en el aula puede llegar un mes
  con una variable rota que la tabla no conoce. Variante: detectarlas dentro de `make_features` comparando cada
  mes contra los anteriores (casi todo cero o vacío cuando antes no) y pasarlas a NA. Se mide sobre todo con la data
  modificada. Fuente: notebook de drifting de la cátedra (z701).
- **P2. Deriva de montos.** Informe (`tools/fase0.py`): montos que se corren más de 25% contra los meses anteriores
  (inflación, aguinaldo). Alimenta E6 y E17.
- **P3. Fuga.** `tools/fuga.py` cada vez que cambie `make_features`. Fuente: *The Hidden Costs of 99% Accuracy*.
- **P4. BAJA+2 por mes.** Tasa por mes en 2019 y razón contra dos meses antes. Alimenta E7.

## Básicos

- **B1. Clase objetivo.** La base es multiclase con BAJA+1 a medio peso. Variantes: BAJA+1 a peso 0,25 y 1; binaria
  BAJA+1 ∪ BAJA+2; binaria sólo BAJA+2. En la validación de 2019 del lab, binaria BAJA+1 ∪ BAJA+2 le ganaba a sólo
  BAJA+2.
- **B2. Lags por mes calendario.** La base toma el "mes anterior" como la fila anterior del cliente: si faltó un mes,
  el lag es de dos meses atrás. Variante: lag del mes calendario (vacío si no estuvo).
- **B3. Submuestreo de CONTINUA.** 1,0 / 0,5 / 0,2 (base) / 0,1.
- **B4. Usar el mes con sólo BAJA+1.** El escenario da las BAJA+1 de m − 1 y la base no las usa. Variantes: sumar m − 1
  al entrenamiento con BAJA+1 como positivas y el resto como negativas; sumarlo sólo para el especialista. Fuente: la
  simulación del aula (estrategias `rank` y `rank_sinparcial`).
- **B5. Qué aporta cada pieza.** Sin especialista / sin segunda etapa / sólo el general. Si una pieza no suma en estos
  datos, la regla de simplicidad la saca.

## Prioridad 1: historia y ventanas

Fuentes: *A Rolling-Window Framework for Churn Prediction*; *How Early Is Early Enough?*; *Predicting Customer Churn:
Extreme Gradient Boosting with Temporal Data*.

- **E1. Meses de entrenamiento dentro de la ventana.** El general usa 3; variantes 1, 2 y 4 (los 4 con clase completa
  que deja la regla). Escalar `min_data_in_leaf` con las filas.
- **E1b. Romper la regla de 6 meses (sólo si Emmanuel la aprueba).** Todos los meses de 2019 disponibles. Sirve para
  saber cuánto cuesta la regla, no para entregarlo.
- **E2. Sin meses rotos.** Sacar del entrenamiento los meses con variables rotas (201905, 201910) en lugar de pasarlas a
  NA.
- **E3. Peso por antigüedad del mes.** Peso = 0,5^(edad / h), h ∈ {1, 2, 3}: los meses recientes pesan más.
- **E4. Largo de la ventana de variables.** Lags 1..k con k ∈ {2 (base), 3, 6}; promedio, mínimo, máximo y tendencia
  sobre 3 y 6 meses. Reportar la curva entera (ganancia contra k): el paper muestra que el codo depende del diseño. Las
  variables pueden mirar meses anteriores a la ventana de entrenamiento.
- **E5. Memoria larga.** Caída contra el máximo de 6 y 12 meses (saldo, transacciones, consumo) y meses desde el último
  sueldo cobrado.

## Prioridad 1: estacionalidad y drifting

Fuentes: *Seasonal false alarms in customer churn and decline early-warning systems*; notebook de drifting de la
cátedra (z701); la simulación del aula.

- **E6. Deltas interanuales: no entra.** x − lag12 contra el mismo mes del año anterior. Con 2019 no se puede validar (no
  hay 2018), y meterla en el modelo sin validarla sería decidir con 2020.
- **E7. Corte con la tasa reciente.** Estimar cuántos BAJA+2 va a tener el mes con la tasa de los meses anteriores (P4)
  y fijar los envíos con eso (`corte` en `train.py`). Con un solo año no hay estacionalidad que estimar.
- **E17. Rank por mes de todos los montos.** La base pasa a percentil 14 montos elegidos para 2021. Variantes: todos
  los montos; todo (como la estrategia `rank` del aula); rank antes de calcular la historia (`rank_previo`), que no se
  mueve si cambia la escala entre meses. Se mide sobre todo con la data modificada.
- **E20. Sacar variables que derivan.** Sin mirar la clase: PSI de cada variable entre la ventana de entrenamiento y el
  mes a puntuar; las de PSI alto y poca importancia no entran. Es el control que enseña la cátedra, llevado al modelo.

## Prioridad 2: hiperparámetros

Fuentes: *Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch*; *Agentic Search
Spaces for Tabular Machine Learning*; *When Is an LLM Worth It for Hyperparameter Optimization?*

Regla: el agente no mueve hiperparámetros a mano. Diseña el espacio, Optuna busca (`tools/hpo.py`) y el ganador se
corre como experimento. Como el único lugar para medir es el test, la búsqueda sobreajusta: pocos trials, y mirar si la
ganancia se sostiene con la data modificada.

- **E8. Espacio ampliado.** El agente escribe `hpo/espacio_ampliado.json` con una línea de justificación por parámetro
  (además de los 4 de base: rondas, bagging, pos/neg_bagging, L1, L2, min_gain, max_depth, max_bin, extra_trees,
  path_smooth, submuestreo). Comparar contra el espacio base con los mismos trials.
- **E9. Arranque.** Primer trial = la configuración vigente; TPE con y sin arranque al mismo presupuesto.
- **E10. Híbrido agente + TPE.** Cada 10 trials el agente lee el estado y propone hasta 2; comparar contra TPE puro.

## Prioridad 3: variables generadas por agentes

Fuente: *Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation (MALMAS)*.

- **E11. Generación por roles con memoria.** Roles temporal, interanual, agregados por grupo e interacciones; 2 por
  ronda, 4 rondas. Una variable entra si supera al mejor de 20 canaritos y suma en el ensamble. Cada intento va a
  `features_probadas.tsv`. Expectativa baja (en el paper, +0,006 de AUC).
- **E12. Agregados por grupo sin fuga.** Tasa histórica de BAJA+2 del grupo del cliente (bandas de transacciones ×
  antigüedad), con meses con clase hasta m − 2.
- **E21. Variables redundantes.** Diagnóstico del paper: si una variable se explica en más de 95% (R²) con otras, no
  aporta y ensucia la importancia. Sacarlas y ver si la ganancia se mantiene con menos columnas. Fuente: *The Hidden
  Costs of 99% Accuracy*.

## Prioridad 3: corte, costos y ensambles

Fuentes: *Calibrated, cost-sensitive, explainable ensemble learning for customer churn prediction in retail banking*
(Frontiers); *Profit Driven Decision Trees for Churn Prediction* (ProfTree); *Data-Driven Telecom Marketing
Optimization*; la simulación del aula.

- **E13. Corte por probabilidad calibrada.** Calibrar P(BAJA+2) con isotónica en el último mes con clase (el
  submuestreo y los pesos la descalibran) y estimular a quien tenga P > 27.500 / 1.100.000 = 2,5%. Variantes: calibrar
  con 1, 2 y 3 meses.
- **E13b. Corte por la proporción aprendida.** La proporción de envíos que maximizó la ganancia en los últimos meses con
  clase, puntuados fuera de muestra (la regla `frac` del script del aula).
- **E14. Tamaño del semillerío.** Curva de la ganancia con 5, 10, 20 y 40 semillas.
- **E18. Corte por segmento.** Una proporción de envíos para el grupo riesgoso (internet == 0) y otra para el resto,
  cada una aprendida de los meses con clase.
- **E19. Pesos por costo en lugar de submuestreo.** Entrenar sin submuestreo y con peso de BAJA+2 según la ganancia
  (1.072.500 / 27.500 ≈ 39) o una fracción; es el "class weighting" del paper de banca y la idea de ProfTree de meter la
  ganancia en el entrenamiento.
- **E22. Ensamble de modelos distintos.** Promediar (por percentil) la base con los modelos binarios de la simulación
  del aula (`historia`, `rank`). El paper de banca encuentra que el ensamble por votación casi empata con el mejor
  modelo; acá los modelos ven variables distintas.

## Prioridad 4: proceso del loop

Fuentes: *Can We Predict Before Executing Machine Learning Agents? (FOREAGENT)*; *Recovering Wasted Compute in
Autoresearch Agents*.

- **E15. Ordenar la cola por predicción.** Antes de cada tanda el agente ordena las ideas pendientes por la ganancia que
  espera y anota la predicción en la hipótesis. Sirve para ordenar, no para descartar (el paper acierta 61,5% de los
  pares).
- **E16. Volver atrás.** Con 5 descartes seguidos, retomar la segunda mejor rama en vez de seguir desde el vigente.

## Para el aula (202007, 202008, 202009)

En el aula se simula estar en cada mes y se implementan los cambios que hagan falta con lo que llega. No se puede
cambiar el modelo por lo que pasó en 2020; sí reaccionar a la data que llega. Lo que se prepara ahora, probado sobre la
data modificada simulada de 2019:

- **Controles de calidad del mes que llega** (nulos, ceros, PSI contra la ventana de entrenamiento y contra el mes
  anterior), con las variables rotas pasadas a NA (P1).
- **Defensas contra el drifting** que el modelo ya traiga definidas en 2019 (E17, E20).
- **El corte re-estimado** con los meses que se van completando (E13, E13b).
- **El tablero** para decidir en el momento: alertas de calidad, distribución del score contra los meses anteriores y
  qué cambia la lista con cada alternativa.

## Propuestas de las compañeras

Llegan desde la página (formulario → issue con la etiqueta `propuesta`). Emmanuel las aprueba poniéndoles la etiqueta
`aprobada`; recién ahí entran a la cola, se corren con este mismo protocolo y el resultado se comenta en el issue.

## No hacer

- Redes profundas o modelos de series de tiempo: en churn tabular el GBDT les gana (*ChurnNet*).
- Modelos fundacionales tabulares (*TabFM-Auto*): no escalan a 4,7 millones de filas y no están instalados.
- Explicaciones SHAP para el CRM (*Explainable AI for churn*): no cambian la ganancia.
- Simuladores de clientes con modelos de lenguaje (*Simulated Customers Never Walk Away*): otro problema.
- Ganancia por cliente (*e-Profits*): acá todos los aciertos valen lo mismo.
- Variables que identifiquen el mes; mover hiperparámetros a mano; elegir semillas o el corte mirando el test.

## Fuentes

- A Rolling-Window Framework for Churn Prediction and Behavioral Driver Identification — https://arxiv.org/abs/2606.06776
- How Early Is Early Enough? Design-Dependent Observation-Window Sufficiency in Subscription Churn Prediction — https://arxiv.org/abs/2607.00473
- Seasonal false alarms in customer churn and decline early-warning systems — https://arxiv.org/abs/2608.18174
- Agentic Search Spaces for Tabular Machine Learning — https://arxiv.org/abs/2609.16309
- The Hidden Costs of 99% Accuracy: A Trustworthiness Audit of the Telco Customer Churn Benchmark — https://arxiv.org/abs/2610.00118
- Calibrated, cost-sensitive, explainable ensemble learning for customer churn prediction in retail banking — https://doi.org/10.3389/frai.2026.1889639
- When Is an LLM Worth It for Hyperparameter Optimization? — https://arxiv.org/abs/2606.21641
- Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch — https://arxiv.org/abs/2603.24647
- Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data — https://arxiv.org/abs/2604.20261
- Data-Driven Telecom Marketing Optimization — https://arxiv.org/abs/2607.10260
- ChurnNet: A Optimized Modern AI for Churn Prediction — https://arxiv.org/abs/2606.00169
- Can We Predict Before Executing Machine Learning Agents? — https://arxiv.org/abs/2601.05930
- Recovering Wasted Compute in Autoresearch Agents — https://arxiv.org/abs/2608.10424
- TabFM-Auto: Self-Evolving Pipelines for Tabular Foundation Models — https://arxiv.org/abs/2609.37989
- Explainable Artificial Intelligence for Customer Churn Prediction in Telecommunications — https://arxiv.org/abs/2608.26151
- Simulated Customers Never Walk Away — https://arxiv.org/abs/2606.20708
- Profit Driven Decision Trees for Churn Prediction (ProfTree) — https://arxiv.org/abs/1712.08101
- e-Profits: customer-level profit for churn — https://arxiv.org/abs/2507.08860
- Predicting Customer Churn: Extreme Gradient Boosting with Temporal Data — https://arxiv.org/abs/1802.03396
- Notebook de drifting y calidad de datos de la cátedra — https://github.com/dmecoyfin/dmeyf2026/blob/main/monday/z701_drifting.ipynb
