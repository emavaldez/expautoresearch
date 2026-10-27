# Búsqueda hpo01

Base: E019 | trials completos: 12 | semillas: [800311, 800329, 800333]

## Mejores 5

| trial | ganancia | origen | parámetros |
|---|---|---|---|
| 1 | 258,5M | tpe | `{"learning_rate": 0.015795386074823305, "num_leaves": 254, "min_data_in_leaf": 72, "feature_fraction": 0.5168676217650697, "lambda_l2": 0.0032872028799478836, "num_rounds": 667, "submuestreo": 0.1320012584263648}` |
| 11 | 257,2M | tpe | `{"learning_rate": 0.024018174943262697, "num_leaves": 255, "min_data_in_leaf": 616, "feature_fraction": 0.679906196589371, "lambda_l2": 0.0011391200110277668, "num_rounds": 810, "submuestreo": 0.3278824681155621}` |
| 3 | 256,6M | tpe | `{"learning_rate": 0.011247822366406144, "num_leaves": 160, "min_data_in_leaf": 56, "feature_fraction": 0.5967710968564564, "lambda_l2": 0.030753308190665907, "num_rounds": 978, "submuestreo": 0.3616291400373393}` |
| 12 | 256,4M | tpe | `{"learning_rate": 0.023757937354556415, "num_leaves": 254, "min_data_in_leaf": 651, "feature_fraction": 0.6636598362890443, "lambda_l2": 0.0015152591969619482, "num_rounds": 811, "submuestreo": 0.3085755985324018}` |
| 2 | 255,2M | tpe | `{"learning_rate": 0.04379323831197828, "num_leaves": 63, "min_data_in_leaf": 94, "feature_fraction": 0.8140902923802459, "lambda_l2": 12.006893707043021, "num_rounds": 577, "submuestreo": 0.1728393138822684}` |

## Parámetros del mejor en el borde del rango

- num_leaves en el máximo del rango (254)

## Importancia (fANOVA, orientativa con pocos trials)

- num_leaves: 0.66
- feature_fraction: 0.15
- submuestreo: 0.06
- num_rounds: 0.06
- min_data_in_leaf: 0.04
- learning_rate: 0.02
- lambda_l2: 0.01

Arranque (configuración base): 253,5M
