# test script for analyses and modeling of 2p calcium imaging data analyses
# cross registration of 2
# learning script
import copy
from json.tool import main
import os 
import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

summaryROI = 'E:\Alice_2p\summaryROI'
keyword    = []

def find_mat(path, keyword):
    path = Path(path)
    if path.is_file() and path.name == "output_behaviorConcat.mat":
        return path
    if path.is_dir():
        for f in path.glob("**/output_behaviorConcat.mat"):
            return f
    raise FileNotFoundError(f"no output_behaviorConcat.mat found in {path}")

def load_behavior(path):
    # load summaryROI
    """Per-frame behavior fields from output_behaviorConcat.mat."""
    d = loadmat(path) 
    d = d.get("concatenatedBehavior", d)
    T = self.dff.shape[1]
    vecs = {}
    for k, v in d.items():
        try:
            a = np.asarray(v)
            if a.ndim == 2 and 1 in a.shape:
                a = a.ravel()
            if a.ndim == 1 and a.dtype.kind in "fiub" and a.size > 1:
                vecs[k] = a.astype(np.float64)
        except Exception:
                pass
        if not vecs:
            raise ValueError("no per-frame numeric fields found")
        # per-frame fields share one length: the most common vector length
        lens = [a.size for a in vecs.values()]
        n = max(set(lens), key=lens.count)
        vecs = {k: a for k, a in vecs.items() if a.size == n}
        if n != T:
            vecs = {k: a[np.minimum(np.arange(T), n - 1)] for k, a in vecs.items()}
        if all(f in vecs for f in BALL):
            vecs[BALL_SPEED] = sum(np.abs(vecs[f]) for f in BALL)
        self.behav, self.behav_file = vecs, str(path)

if __name__ == "__main__":
    main()