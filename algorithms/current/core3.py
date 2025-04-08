import random
from typing import List, Tuple

from algorithms.utils import val


def search(
    soln_init: List[int],
    tabu_tenure: int,
    iter_max: int = 100,
) -> Tuple[int, List[int]]:
    tabu_list: List[Tuple[int, int]] = []
    soln_curr: List[int] = soln_init.copy()
    soln_best: List[int] = soln_init.copy()
    soln_best_tracker: List[int] = []

    for iter_ctr in range(iter_max):
        nbhr, move = generate_neighbor(soln_curr, tabu_list)

        if val(nbhr) < val(soln_best):
            soln_best = nbhr.copy()
        soln_best_tracker.append(val(nbhr))
        soln_curr = nbhr.copy()

        tabu_list.append(move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker


def generate_neighbor(
    soln: List[int], tabu_list: List[Tuple[int, int]]
) -> Tuple[List[int], Tuple[int, int]]:
    while True:
        soln_mod: List[int] = soln.copy()
        n: int = len(soln)

        i, j = random.sample(range(n), 2)
        move: Tuple[int, int] = (soln_mod[i], soln_mod[j])
        soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]

        if move not in tabu_list:
            return soln_mod, move
