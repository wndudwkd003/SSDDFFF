# utils/strategy.py

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return float(np.mean(pred[pred > t]))
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return float(np.mean(pred[pred < 0.2]))
    else:
        return float(np.mean(pred))
