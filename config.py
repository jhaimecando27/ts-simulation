import pandas as pd
import random
import numpy as np
from typing import Dict, List

runs: int = 5
iter_max: int = 100
dms: Dict[str, np.ndarray] = {}
soln_inits: Dict[str, List[int]] = {}
pois: List[str] = ["40", "80", "160"]
tenures: List[int] = [10, 20, 30]
distance_matrix: np.ndarray = np.array([])

for poi in pois:
    csv: str = f"data/input/poi_{poi}.csv"
    distance_matrix = pd.read_csv(csv).values[:, 1:]
    soln_init: List[int] = list(range(distance_matrix.shape[0]))
    random.shuffle(soln_init)
    dms.update({poi: distance_matrix})
    soln_inits.update({poi: soln_init})
