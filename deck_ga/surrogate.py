"""
Surrogate Model — Prediksi win rate deck tanpa simulasi.

Gunakan RandomForestRegressor untuk memprediksi fitness dari fitur deck.
Ini 100-1000x lebih cepat dari evaluasi dengan game engine.

Fase 2: Digunakan setelah terkumpul cukup data training dari evaluasi nyata.
"""
import os
import pickle
import numpy as np
from typing import Optional

from . import config
from .feature_extractor import FIXED_DIM, extract_deck_features
from .card_db import CardDB


class SurrogateModel:
    """
    Prediktor win rate — belajar dari hasil evaluasi nyata.

    Train: fitur deck → actual win rate (dari evaluator)
    Predict: fitur deck → estimated win rate
    """

    def __init__(self):
        self.model = None
        self.X = []  # Feature vectors
        self.y = []  # Win rates
        self._is_trained = False

    def add_observation(self, card_ids: list[int], win_rate: float, db: CardDB):
        """Tambahkan satu data point (deck → win rate)."""
        features = extract_deck_features(card_ids, db)
        self.X.append(features)
        self.y.append(win_rate)

    def add_batch(self, decks: list, win_rates: list[float], db: CardDB):
        """Tambahkan batch data points."""
        for deck, wr in zip(decks, win_rates):
            self.add_observation(deck.card_ids, wr, db)

    def train(self):
        """Train surrogate model dari data yang terkumpul."""
        if len(self.X) < 20:
            print(f"[Surrogate] Data terlalu sedikit ({len(self.X)}), skip training")
            return

        X = np.array(self.X)
        y = np.array(self.y)

        try:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=1,
            )
            self.model.fit(X, y)
            self._is_trained = True
            score = self.model.score(X, y)
            print(f"[Surrogate] Trained: R²={score:.3f}, data={len(X)}")
        except ImportError:
            print("[Surrogate] sklearn tidak tersedia, gunakan fallback")
            self._is_trained = False

    def predict(self, card_ids: list[int], db: CardDB) -> Optional[float]:
        """Prediksi win rate dari deck."""
        if not self._is_trained or self.model is None:
            return None

        features = extract_deck_features(card_ids, db).reshape(1, -1)
        pred = self.model.predict(features)[0]
        return float(np.clip(pred, 0.0, 1.0))

    def predict_batch(self, decks: list, db: CardDB) -> list[Optional[float]]:
        """Prediksi win rate untuk banyak deck."""
        if not self._is_trained or self.model is None:
            return [None] * len(decks)

        X = np.array([extract_deck_features(d.card_ids, db) for d in decks])
        preds = self.model.predict(X)
        return [float(np.clip(p, 0.0, 1.0)) for p in preds]

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def save(self, path: str = None):
        """Save model ke disk."""
        if path is None:
            path = os.path.join(config.CHECKPOINT_DIR, "surrogate.pkl")
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            print(f"[Surrogate] Saved to {path}")

    def load(self, path: str = None):
        """Load model dari disk."""
        if path is None:
            path = os.path.join(config.CHECKPOINT_DIR, "surrogate.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self._is_trained = True
            print(f"[Surrogate] Loaded from {path}")
