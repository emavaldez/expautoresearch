# Espacio ampliado (E8), sobre el vigente E019

Una línea por parámetro: por qué entra y por qué ese rango. Se busca con TPE, 3 semillas del bloque de búsqueda, sobre
la meseta de la curva en 201908-201910 (sólo 2019).

- learning_rate 0,01-0,08 (log): el vigente usa 0,027; más bajo con más rondas suele ganar, más alto acorta el tiempo.
- num_leaves 15-255 (log): 31 en el vigente; con ~600.000 filas de entrenamiento hay lugar para árboles más grandes.
- min_data_in_leaf 50-2.000 (log): 200 en el vigente; con ~2.200 bajas por ventana, hojas muy chicas sobreajustan.
- feature_fraction 0,2-0,9: 0,5 en el vigente; con 1.648 columnas, muchas redundantes, bajarla decorrelaciona árboles.
- lambda_l2 0,001-50 (log): 0 en el vigente; regulariza las hojas con pocas bajas.
- num_rounds 250-1.000: 500 en el vigente; acompaña al learning_rate.
- submuestreo 0,1-0,4: 0,2 en el vigente; B3 mostró que 0,5 pierde, así que el techo queda en 0,4.

Fuera a propósito: max_bin (cambia el binning y el tiempo, poco efecto esperado), bagging (el submuestreo ya
aleatoriza filas) y extra_trees (más ruido con pocas bajas).
