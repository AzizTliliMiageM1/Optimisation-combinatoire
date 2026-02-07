## Architecture ML - Version initiale

Objectif
-------

Proposer une architecture simple pour prédire la qualité attendue d'une
solution de type sac-à-dos afin d'améliorer ultérieurement la métaheuristique
(ex. guider la sélection d'opérateurs ou ajuster dynamiquement des
paramètres).

Choix techniques
-----------------

- Modèle: `MLPRegressor` (réseau de neurones multi-couches) — choix simple,
  efficace pour des relations non-linéaires et rapide à prototyper.
- Mise à l'échelle: `MinMaxScaler` pour normaliser les features avant entraînement.

Features
--------

La version initiale extrait 6 features à partir d'une instance (poids, valeurs,
capacité) :

1. Nombre d'objets (`n_items`)
2. Poids moyen (`avg_weight`)
3. Valeur moyenne (`avg_value`)
4. Variance des poids (`weight_var`)
5. Variance des valeurs (`value_var`)
6. Ratio capacité / somme_des_poids (`capacity_ratio`)

Rôle proposé pour le ML
-----------------------

- Estimer la qualité d'une solution candidate (score) pour prioriser les
  solutions explorées par la métaheuristique.
- Servir d'évaluateur rapide lors d'une phase d'exploitation pour éviter des
  évaluations coûteuses.

Limites et suites
-----------------

Cette version est une base de travail (stubs et features simples). Les
améliorations possibles : features basées sur structure des solutions, autre
modèle (ex: random forest), apprentissage par renforcement pour policy-guided search.
