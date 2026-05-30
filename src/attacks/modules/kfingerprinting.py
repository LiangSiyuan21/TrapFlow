# attacks/modules/kfingerprinting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class KFPConfig:
    n_estimators: int = 500
    k: int = 3
    unanimous: bool = True
    random_state: int = 0
    n_jobs: int = -1


class KFingerprintingRF:
    """
    k-fingerprinting (Hayes & Danezis style):
    - Train a RandomForest on features.
    - Fingerprint of a sample = leaf indices from all trees.
    - kNN in fingerprint space (Hamming distance on leaf-ID vectors).
    """
    def __init__(self, cfg: KFPConfig):
        self.cfg = cfg
        self.rf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        self.X_fp_train: Optional[np.ndarray] = None   # (N, T) leaf ids
        self.y_train: Optional[np.ndarray] = None
        self.num_classes: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KFingerprintingRF":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.rf.fit(X, y)
        self.X_fp_train = self.rf.apply(X).astype(np.int32)  # (N, T)
        self.y_train = y
        self.num_classes = int(np.max(y)) + 1
        return self

    def _hamming_distance(self, fp_query: np.ndarray, fp_train: np.ndarray) -> np.ndarray:
        # fp_query: (T,), fp_train: (N, T)
        return np.sum(fp_train != fp_query[None, :], axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.X_fp_train is not None and self.y_train is not None and self.num_classes is not None, \
            "Call fit() before predict_proba()."

        X = np.asarray(X, dtype=np.float32)
        fp_q = self.rf.apply(X).astype(np.int32)  # (M, T)

        M = fp_q.shape[0]
        C = self.num_classes
        probs = np.zeros((M, C), dtype=np.float32)

        k = int(self.cfg.k)
        for i in range(M):
            d = self._hamming_distance(fp_q[i], self.X_fp_train)
            nn_idx = np.argsort(d)[:k]
            nn_labels = self.y_train[nn_idx]

            if self.cfg.unanimous:
                # unanimous: only predict a monitored label if all k agree
                if np.all(nn_labels == nn_labels[0]):
                    probs[i, nn_labels[0]] = 1.0
                else:
                    # 如果你是 open-world，通常这里应该判为 unmon class
                    # 但你代码里 unmon label 是 self.nmc (最后一类)，
                    # 我们在外层 wrapper 里会把它映射进去。
                    # 这里先给全零，外层再处理。
                    pass
            else:
                # majority vote
                for lb in nn_labels:
                    probs[i, lb] += 1.0
                probs[i] /= float(k)

        return probs
