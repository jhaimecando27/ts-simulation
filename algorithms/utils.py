import random
import numpy as np
from typing import List, Tuple, Optional

import config


def neighborhood(
    soln: List[int], tabu_list: List[List[int]]
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    nbhd: List[List[int]] = []
    moves: List[Tuple[int, int]] = []

    n: int = len(soln)
    for i in range(int(n / 2)):
        for j in range(i + 1, int(n / 2)):
            soln_mod: List[int] = soln.copy()
            soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]

            if (soln[i], soln[j]) in tabu_list:
                continue
            nbhd.append(soln_mod)
            moves.append((soln[i], soln[j]))
    return nbhd, moves


def val(soln: List[int]) -> int:
    value: int = 0
    n: int = len(soln)
    for i in range(n):
        poi_first: int = soln[i]
        poi_second: int = soln[(i + 1) % n]
        value += config.dms[str(len(soln))][poi_first][poi_second]
    return value


def best_admissible_soln(
    nbhd: List[List[int]],
    moves: List[Tuple[int, int]],
    tabu_list: List[List[int]],
    soln_best: List[int],
) -> Tuple[Optional[List[int]], Optional[Tuple[int, int]]]:
    val_best: int = float("inf")
    nbhr_best: Optional[List[int]] = None
    move_best: Optional[Tuple[int, int]] = None

    for idx, nbhr_curr in enumerate(nbhd):
        val_curr: int = val(nbhr_curr)

        if moves[idx] not in tabu_list or val_curr < val_best:
            if val_curr < val_best:
                val_best = val_curr
                nbhr_best = nbhr_curr
                move_best = moves[idx]

    return nbhr_best, move_best
