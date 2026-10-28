# Protocolo de los experimentos (preregistro)

Escrito antes de correr. Lo que cambie después queda registrado abajo, en "Cambios al protocolo", con la fecha y el
motivo.

## Pregunta

¿Qué cambios sobre el baseline de la cátedra mejoran la predicción de BAJA+2 en 2019, medida en un mes que nunca se usó
para decidir (201910)? Y, como pregunta de método: ¿un loop de autoresearch al estilo de Karpathy (un agente que prueba
cambios de a uno y se queda sólo con los que mejoran en validación) llega a un modelo mejor que el baseline?

## Datos

- Dataset de la Segunda Competencia, sólo 2019. La clase se calcula por presencia con los datos hasta 201912, así que
  201911 sólo tiene BAJA+1 y 201912 no tiene clase. Esos dos meses quedan para el tablero y **no se usan** en los
  experimentos.
- **Entrenamiento: 201903, 201904, 201905 y 201906**, completos, sin submuestreo.
- **Validación: 201908.** Ordena las búsquedas de hiperparámetros y decide qué se queda en el loop.
- **Test: 201910.** Sólo para la comparación final. No se usa para elegir nada.
- 201907 y 201909 entran sólo como historia (los lags de 201908 y 201910).

## Baseline (la referencia de la cátedra)

- Variables: las 152 crudas, más lag 1 y lag 2 y delta 1 y delta 2 de cada una.
- Clase: binaria, positivos BAJA+1 o BAJA+2 (la "clase_binaria2" del curso). Se mide siempre contra BAJA+2.
- Modelo: LightGBM binario.
- Búsqueda: Optuna (TPE), 40 trials, sobre `num_leaves` (8 a 512), `learning_rate` (0,01 a 0,2), `min_data_in_leaf`
  (20 a 5.000), `feature_fraction` (0,1 a 1) y `bagging_fraction` (0,3 a 1), con `bagging_freq = 1` fijo y las rondas
  por early stopping en validación (hasta 3.000, paciencia 100). Ordena la **AUC en 201908**.
- Modelo final: los mejores parámetros, 10 semillas, promedio de probabilidades.

## Cómo se mide (igual para todos)

| | Métrica | Por qué |
|---|---|---|
| Primaria | **AUC** en 201910 | La que pide la cátedra: calidad del orden de todos los clientes. |
| Secundaria | **Ganancia meseta** en 201910 | La plata que da el orden alrededor del mejor corte (ver abajo). |
| Informativas | Ganancia de la lista (3,3% de los clientes), AUC parcial (falsos positivos hasta 5%), BAJA+2 entre el 1% de arriba | La plata con un corte decidido sin mirar la clase, y la calidad del orden sólo en la zona de la lista. |

**Ganancias.** Se ordenan los clientes del mes de mayor a menor probabilidad y se los estimula de a uno: cada BAJA+2
suma 1.072.500 y cada uno que no lo es resta 27.500. La ganancia acumulada sube mientras se acierta y después baja.

- *Ganancia máxima:* el pico de esa curva. Es optimista y ruidosa: el corte se elige mirando la respuesta y un acierto
  de más mueve 1,1 M.
- *Ganancia meseta:* el pico de la curva promediada en ventanas de 401 envíos. Mide cuánto vale el orden cerca del mejor
  corte, con menos suerte. Sirve para comparar modelos; no dice dónde cortar.
- *Ganancia de la lista:* la de la lista con un corte decidido sin mirar la clase (el 3,3% del mes, la regla del aula).
  Es lo que de verdad se cobraría: mezcla la calidad del orden con el acierto del corte.

**Incertidumbre.** Cada experimento se compara con el baseline **pareado por semilla** (las mismas 10 semillas): media
de la diferencia, error estándar y Wilcoxon. Y con **bootstrap por clientes** de 201910 (1.000 remuestreos) sobre el
ensamble: intervalo de 95% de la diferencia de AUC y de ganancia meseta.

## Regla de decisión (fijada de antemano)

- **Para cada hipótesis (contra el baseline, en 201910):** se considera que mejora si la diferencia de AUC es positiva,
  su intervalo de 95% por bootstrap no incluye el cero y la diferencia pareada por semilla supera 2 errores estándar.
  Como son varias hipótesis, los p-valores se corrigen con Holm. La ganancia se informa siempre, pero no decide.
- **En el loop de autoresearch (en 201908):** un cambio se queda si la AUC de validación del ensamble sube más de 0,001
  y la diferencia pareada por semilla supera 2 errores estándar; si no, se deshace. A igual AUC gana lo más simple.

## Hipótesis

Cada una es un solo cambio sobre el baseline (variable independiente). Todo lo demás queda fijo: datos, meses,
semillas, búsqueda con el mismo presupuesto de 40 trials ordenada por AUC en 201908, y métricas. Para cada una:
H0 = el cambio no mejora la AUC en 201910; H1 = la mejora, en la dirección de la predicción.

| Id | Hipótesis (H1) | Predicción | Por qué | Fuente |
|---|---|---|---|---|
| H1 | Entrenar como multiclase (CONTINUA / BAJA+1 / BAJA+2) mejora a la binaria | sube poco | separa al que se va el mes que viene del que se va en dos | Primera competencia |
| H1b | Entrenar con BAJA+2 sola como positivo empeora | baja | la mitad de los ejemplos de baja | Curso; lab de 2019 |
| H2 | Las variables de la primera competencia (razones, ventana de 3 meses, tendencia y promedio de la cátedra) mejoran | sube | resumen la historia en forma que el árbol no arma solo | Primera competencia; cátedra |
| H3 | Pasar los montos a percentil por mes con cero fijo mejora | igual o sube poco | la inflación corre los cortes en pesos entre los meses de entrenamiento y el de test | Notebook de drifting de la cátedra |
| H4 | Pasar a NA las variables rotas (201905 y 201910) mejora | sube poco | un cero falso parece un cliente que dejó de operar | Notebook de calidad de datos de la cátedra |
| H5 | Submuestrear CONTINUA al 20% no empeora y es más rápido | igual | las bajas son 1%: casi toda la información está en ellas | Primera competencia |
| H6 | Entrenar con más meses (201901 a 201906) mejora | sube | más ejemplos de baja; los primeros meses no tienen lags | A Rolling-Window Framework for Churn Prediction |
| H7 | Un especialista para el grupo riesgoso (internet = 0) mejora | sube poco o igual | ese grupo tiene ocho veces más bajas y el general lo promedia | Primera competencia |
| H8 | Un especialista en la franja dudosa (segunda etapa con cross-fitting) mejora la ganancia sin bajar la AUC | AUC igual, ganancia sube poco | reordena sólo donde se decide la lista | Primera competencia |
| H9 | Optimizar los hiperparámetros por ganancia en lugar de AUC da más ganancia en test | ganancia sube, AUC igual o baja | la AUC premia el orden de clientes que nunca se estimulan | Pregunta propia |
| H10 | Un espacio de búsqueda ampliado por el agente (regularización, rondas, submuestreo, max_bin) mejora al espacio de la cátedra con el mismo presupuesto | sube poco | | Agentic Search Spaces for Tabular ML |
| H11 | El loop de autoresearch (cambios en cadena elegidos en 201908) supera al baseline y a cualquier cambio solo | sube | combina lo que suma | Karpathy, autoresearch |

## Cómo corre (autoresearch)

Igual que el repositorio de Karpathy: un arnés fijo que el agente no toca (`prepare.py`, `config.py`: datos, meses,
métricas), un solo archivo que el agente modifica (`train.py`), instrucciones del loop en `program.md`, presupuesto
de tiempo fijo por corrida, y todo registrado: el código exacto de cada corrida, su log, su resultado y, si se
descartó, por qué. Fase A: cada hipótesis H1 a H10 contra el baseline. Fase B: el loop (H11), que arranca del baseline y
encadena cambios elegidos sólo en 201908.

## Controles contra el autoengaño

- 201910 no se usa para elegir nada (ni hiperparámetros, ni cambios, ni corte). Se lee para informar.
- Mismo presupuesto de búsqueda para todos: un cambio no gana por tener los hiperparámetros mejor ajustados.
- Control de fuga (`tools/fuga.py`): las variables de un mes no cambian si se agregan meses futuros.
- Todo queda registrado, también lo que no anduvo: no hay hipótesis que se esconda por mala.
- Corrección por comparaciones múltiples (Holm).

## Limitaciones (para el video)

- Un solo mes de test con unas 630 BAJA+2: alcanza para diferencias de AUC de unas milésimas, no menos, y las de
  ganancia tienen intervalos anchos.
- Un solo mes de validación: la búsqueda puede sobreajustar a 201908.
- Todo es 2019: no hay pandemia ni inflación fuerte. Cómo se comporta en 2020 es la otra entrega (el aula).
- 40 trials por búsqueda: un espacio grande puede quedar mal explorado.
- Los meses de entrenamiento están fijos por el baseline (salvo en H6).

## Cambios al protocolo

- (ninguno todavía)
