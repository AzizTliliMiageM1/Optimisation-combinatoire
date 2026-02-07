"""
src/ml_model.py
Version 1.0 - MLKnapsackModel

Interface minimale pour entraîner/prédire la qualité d'une solution
pour un problème type sac-à-dos (knapsack). Ceci est une version
initiale (stubs / minimal implementation) demandée pour le commit 1.
"""
from typing import List, Dict, Any

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


class MLKnapsackModel:
    """Version 1.0

    - Classe simple exposant `fit(X, y)` et `predict(X)`.
    - Extraction de 6 features basiques à partir d'une instance.
    - Mise à l'échelle Min-Max et modèle `MLPRegressor`.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), random_state=42, max_iter=200):
        self.scaler = MinMaxScaler()
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                  random_state=random_state,
                                  max_iter=max_iter)
        self.is_fitted = False

    @staticmethod
    def _extract_features_from_instance(instance: Dict[str, Any]) -> np.ndarray:
        """
        Calcule 6 features à partir d'une instance du sac-à-dos.

        instance doit être un dict contenant au minimum :
        - 'weights': list[float]
        - 'values': list[float]
        - 'capacity': float

        Features retournées (shape (6,)) :
        1. n_items
        2. avg_weight
        3. avg_value
        4. weight_var
        5. value_var
        6. capacity_ratio (capacity / sum(weights))
        """
        weights = np.asarray(instance.get("weights", []), dtype=float)
        values = np.asarray(instance.get("values", []), dtype=float)
        capacity = float(instance.get("capacity", 0.0))

        if weights.size == 0:
            return np.zeros(6, dtype=float)

        n = float(weights.size)
        avg_w = float(weights.mean())
        avg_v = float(values.mean())
        var_w = float(weights.var())
        var_v = float(values.var())
        capacity_ratio = float(capacity / (weights.sum() + 1e-9))

        return np.array([n, avg_w, avg_v, var_w, var_v, capacity_ratio], dtype=float)

    def _prepare_X(self, X: List[Dict[str, Any]]) -> np.ndarray:
        """Transforme une liste d'instances en matrice de features (n_samples, 6)."""
        feats = [self._extract_features_from_instance(x) for x in X]
        return np.vstack(feats)

    def fit(self, X: List[Dict[str, Any]], y: List[float]):
        """Entraîne le pipeline scaler + MLPRegressor.

        X: liste d'instances (dicts)
        y: cible numérique (qualité / score attendu)
        """
        Xmat = self._prepare_X(X)
        Xs = self.scaler.fit_transform(Xmat)
        self.model.fit(Xs, np.asarray(y, dtype=float))
        self.is_fitted = True

    def predict(self, X: List[Dict[str, Any]]) -> np.ndarray:
        """Prédit la valeur cible pour une liste d'instances.

        Retourne un tableau numpy 1D des prédictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        Xmat = self._prepare_X(X)
        Xs = self.scaler.transform(Xmat)
        return self.model.predict(Xs)


if __name__ == "__main__":
    # Petit test local / demonstration rapide
    inst = {
        'weights': [2, 3, 5, 7],
        'values': [3, 4, 8, 10],
        'capacity': 10
    }
    model = MLKnapsackModel()
    X = [inst, inst]
    y = [12.0, 11.5]
    # Les appels suivants sont des stubs utilisables pour tests locaux
    model.fit(X, y)
    preds = model.predict(X)
    print('preds:', preds)
