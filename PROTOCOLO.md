# Protocolo de los experimentos (preregistro)

Escrito antes de correr. Lo que cambie después queda registrado abajo, en "Cambios al protocolo", con la fecha y el
motivo. La lista completa de hipótesis, con su predicción y su fuente, está en `hipotesis.tsv` (y en la página).

## Pregunta

¿Un loop de autoresearch al estilo de Karpathy, que parte del baseline de la cátedra y prueba de a uno los cambios que
salen de los papers, de la primera competencia y de ideas propias, llega a un modelo que predice mejor las BAJA+2 de
2019 en un mes que nunca se usó para decidir (201910)?

**Hipótesis principal (H11).** El modelo final del loop tiene más AUC en 201910 que el baseline. Predicción: entre
+0,003 y +0,008 de AUC y entre +5 y +15 M de ganancia meseta. H0: no la mejora.

Preguntas que la acompañan: qué cambios quedan y de qué fuente vienen; si las ideas de los papers rinden más que las de
la primera competencia; cómo conviene buscar los hiperparámetros; cuánto se sobreajusta el loop a 201908; si el agente
sabe de antemano qué va a andar.

## Datos

- Dataset de la Segunda Competencia, sólo 2019. La clase se calcula por presencia con los datos hasta 201912, así que
  201911 sólo tiene BAJA+1 y 201912 no tiene clase. Esos dos meses quedan para el tablero y **no se usan**.
- **Entrenamiento: 201903, 201904, 201905 y 201906**, completos, sin submuestreo (lo cambian sólo H6, H23 y H5).
- **Validación: 201908** (542 BAJA+2). Ordena las búsquedas, hace el early stopping y decide cada paso del loop.
- **Test: 201910** (632 BAJA+2). Sólo para la comparación final. No se usa para elegir nada.
- 201901, 201902, 201907 y 201909 entran sólo como historia (los lags).

## Baseline (B0, la referencia de la cátedra)

- Variables: las 152 crudas, más lag 1 y lag 2 y delta 1 y delta 2 de cada una (760). `foto_mes` y el número de
  cliente no son variables: identificarían el mes o al cliente.
- Clase: binaria, **BAJA+2 contra el resto** (CONTINUA o BAJA+1).
- Modelo: LightGBM binario.
- Búsqueda: Optuna (TPE), 40 trials, sobre `num_leaves` (8 a 512), `learning_rate` (0,01 a 0,2), `min_data_in_leaf`
  (20 a 5.000), `feature_fraction` (0,1 a 1) y `bagging_fraction` (0,3 a 1), con `bagging_freq = 1` fijo y las rondas
  por early stopping en 201908 (hasta 3.000, paciencia 100). Ordena la **AUC en 201908**.
- Modelo final: los mejores parámetros, 10 semillas, promedio de probabilidades.

## Cómo se mide (igual para todo)

| | Métrica | Qué mide |
|---|---|---|
| Primaria | **AUC** | La calidad del orden de todos los clientes, sin ningún corte. |
| Secundaria | **Ganancia meseta** | La plata que vale el orden en la parte de arriba del ranking, sin la suerte del último cliente. |
| Informativas | Ganancia de la lista (3,3% del mes), AUC parcial (falsos positivos hasta 5%), BAJA+2 entre el 1% de arriba, ganancia máxima | La plata con un corte decidido a ciegas; el orden sólo en la zona de arriba. |

**Ganancias.** Se ordenan los clientes del mes de mayor a menor probabilidad y se los estimula de a uno: cada BAJA+2
suma 1.072.500 y cada uno que no lo es resta 27.500. La ganancia acumulada sube mientras se acierta, se aplana y después
baja: arriba queda una parte plana, la **meseta**, donde mandar mil envíos más o menos casi no cambia la plata.

- *Ganancia máxima:* el punto más alto de la curva. Se elige mirando la respuesta, y cada acierto de más mueve 1,1 M:
  depende de dónde cayó el último BAJA+2. Se informa, no compara.
- *Ganancia meseta:* la altura de la parte plana. Cada punto de la curva se promedia con los 200 envíos de antes y los
  200 de después (401) y se toma el más alto. Es lo que vale ese orden si se corta en cualquier lugar razonable cerca del
  mejor. Sirve para comparar modelos en pesos. **No dice dónde cortar**: dónde está el pico se sabe recién con la clase.
- *Ganancia de la lista:* la del corte que se usaría de verdad, decidido sin mirar la clase (el 3,3% del mes, la regla
  del aula). Mezcla la calidad del orden con el acierto del corte.

**Incertidumbre.** Toda comparación es **pareada por semilla** (las mismas semillas: media de la diferencia, error
estándar y Wilcoxon) y, en 201910, con **bootstrap por clientes** (1.000 remuestreos, los mismos clientes para los dos
modelos) sobre el ensamble: intervalo de 95% y p-valor de la diferencia de AUC y de ganancia meseta.

## Diseño: todo es un loop

Como en el repositorio de Karpathy: un arnés fijo que el agente no toca (`v3/arnes.py`: datos, meses, semillas,
métricas, comparación), un solo archivo que el agente modifica (`v3/train.py`), instrucciones del loop en
`v3/program.md`, y todo registrado (el código exacto de cada paso, su log y su resultado, también lo que se descarta).

1. **Baseline.** `v3/train.py` arranca como B0; se corre su búsqueda y sus 10 semillas. Los mejores parámetros se
   copian a `v3/train.py`.
2. **La cola.** Las ideas del grupo `loop` de `hipotesis.tsv`, cada una un solo cambio, con su predicción escrita antes
   de correr (`pred_auc`: la diferencia de AUC en 201908 que espera el agente). La cola se ordena por esa predicción, de
   mayor a menor (H33). Al vaciarse, el agente propone hasta 10 variables nuevas leyendo lo aprendido (H28).
3. **Cada paso.** Vigente + un cambio, con los parámetros del vigente, rondas por early stopping en 201908, 5 semillas
   (las 5 primeras de las 10). Se compara contra el vigente **en 201908**:
   - un cambio que **agrega** se queda si la AUC del ensamble sube más de 0,001 y la diferencia pareada por semilla supera
     2 errores estándar;
   - un cambio que **saca** algo (simplificación: H5, H13, H23, H31) se queda si la AUC no baja más de 0,0005 y la
     diferencia pareada es mayor que −2 errores estándar;
   - los cambios **pensados para la ganancia** (H8, H14, H30) se quedan si la ganancia meseta de 201908 sube más de 2 M
     con diferencia pareada mayor que 2 errores estándar y la AUC no baja más de 0,0005.

   Si queda, se commitea `v3/train.py` (la historia del modelo queda en git); si no, se deshace. Una idea con variantes
   (H24: tres segmentos) prueba cada una y queda la mejor en 201908.
4. **Final.** El vigente al terminar la cola corre con 10 semillas; después se le vuelve a buscar hiperparámetros con 40
   trials (H25) y se corre de nuevo con 10. Ese es el modelo final; se compara con B0 en 201910.
5. **Ablación (exploratoria).** En 201910, el final sin cada cambio que quedó, de a uno: cuánto aporta cada uno al
   final, para el video.

Aparte del loop, porque no son cambios a `train.py`:

- **Búsqueda** (H9, H10, H15, H35): B0 con otra forma de buscar, los mismos 40 trials y 10 semillas, contra B0 en 201910.
- **Sobre el loop** (H33, H34, H18): si el agente predice bien, si cortar temprano ahorraría cómputo sin perder nada, y
  si más semillas suben el final.
- **Corte** (H36, H37, H38): sobre el modelo final, otras reglas para decidir cuántos estimular. No cambian el orden (la
  AUC es la misma), sólo la ganancia de la lista. Se eligen con 201908 y se informan en 201910.

## Regla de decisión (fijada de antemano)

- **Confirmatorias** (en 201910, corregidas por Holm entre ellas): H11, H25, H9, H10, H15, H35 y H18. Se considera que
  mejora si la diferencia de AUC es positiva, su intervalo de 95% por bootstrap no incluye el cero, la diferencia pareada
  por semilla supera 2 errores estándar y el p-valor de bootstrap ajustado por Holm es menor que 0,05. H9 se decide igual
  pero por ganancia meseta (es lo que predice). La ganancia se informa siempre.
- **Exploratorias:** cada idea de la cola (se decide en 201908 por la regla del loop; su efecto en 201910 sale de la
  ablación), las de corte, H33 y H34. Se informan con su intervalo, rotuladas como exploratorias.

## Las hipótesis

La lista completa (40) está en `hipotesis.tsv`, con la predicción, el porqué y la fuente de cada una. En resumen:

| Grupo | Ideas | De dónde vienen |
|---|---|---|
| La cola del loop (27) | H1, H1b, H2, H3, H4, H5, H6, H7, H8, H12, H13, H14, H16, H17, H19 a H24, H26 a H32 | Papers (Rolling-Window, How Early, Calibrated, ProfTree, e-Profits, Hidden Costs, Simulated Customers, XAI, ChurnNet, TabFM, XGBoost temporal, MALMAS), la primera competencia, la cátedra e ideas propias |
| Búsqueda (5) | H9, H10, H15, H35, H25 | Agentic Search Spaces, When Is an LLM Worth It, Centaur, ProfTree, la primera |
| Sobre el loop (3) | H33, H34, H18 | FOREAGENT, Recovering Wasted Compute, la primera |
| Corte (3) | H36, H37, H38 | Calibrated, Seasonal false alarms, Telecom |

Dos ideas necesitan instalar algo que no está en la Mac y esperan el OK: H29 (`torch`) y H30 (`tabpfn` y sus pesos).

## Controles contra el autoengaño

- 201910 no se usa para elegir nada (ni hiperparámetros, ni cambios, ni corte, ni segmentos). Se lee para informar.
- Mismo presupuesto de búsqueda para el baseline y el final (40 trials cada uno).
- Control de fuga: las variables de un mes no cambian si se agregan meses futuros (comprobado para B0).
- Todo queda registrado, también lo que no anduvo. Las predicciones se escriben antes de correr.
- Corrección por comparaciones múltiples (Holm) en las confirmatorias.

## Limitaciones (para el video)

- Un solo mes de test con 632 BAJA+2: alcanza para diferencias de AUC de unas milésimas, no menos; las de ganancia
  tienen intervalos anchos.
- Un solo mes de validación: el loop y la búsqueda pueden sobreajustar a 201908. Se mide: la diferencia entre lo que el
  loop ganó en 201908 y lo que se ve en 201910.
- El loop depende del orden de la cola: una idea que no suma sola puede sumar después de otra, y al revés.
- En el loop cada cambio se prueba con los parámetros del vigente, y con 5 semillas: un cambio que pide otros
  hiperparámetros puede quedar afuera (H25 lo mide sólo al final).
- Todo es 2019: no hay pandemia. Cómo se comporta en 2020 es la otra entrega (el aula).

## Cambios al protocolo

- 2026-10-28. Antes de correr nada:
  - La clase del baseline pasa a ser **BAJA+2 contra el resto** (antes, BAJA+1 o BAJA+2). Lo de contar las BAJA+1
    queda como idea (H1b).
  - Todo pasa a ser un loop. Antes eran dos fases: cada hipótesis contra el baseline y después el loop.
  - Se suman todos los papers, todas las ideas de la primera competencia y las propuestas del agente (de 13 a 40).
  - Error corregido en el arnés (`prepare.cargar`): el número de cliente se pasaba a float32 antes que a entero, y
    float32 no distingue enteros de más de 16,7 millones. Quedaban 214 números con dos clientes distintos, con la clase
    y los lags mezclados (y 1 a 3 BAJA+2 por mes tapadas). Afectó a los experimentos viejos (E000 a E040), no a la
    herramienta del aula, que convierte el id primero.
