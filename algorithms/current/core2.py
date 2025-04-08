import random
from typing import List, Tuple

import config
from algorithms.utils import val, best_admissible_soln


def search(
    soln_init: List[int],
    tabu_tenure: int,
    iter_max: int = 100,
) -> Tuple[int, List[int]]:
    n_size: int = int(len(soln_init) * 0.1)
    tabu_list: List[Tuple[int, int]] = []
    soln_curr: List[int] = soln_init.copy()
    soln_best: List[int] = soln_init.copy()
    soln_best_tracker: List[int] = []

    for iter_ctr in range(iter_max):
        nbhd: List[List[int]]
        moves: List[Tuple[int, int]]
        nbhd, moves = neighborhood(soln_curr, tabu_list, n_size)

        nbhr_best: List[int]
        move_best: Tuple[int, int]
        nbhr_best, move_best = best_admissible_soln(nbhd, moves, tabu_list, soln_best)

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best.copy()
        soln_best_tracker.append(val(nbhr_best))

        tabu_list.append(move_best)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker


def neighborhood(
    soln: List[int], tabu_list: List[Tuple[int, int]], n_size: int
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    nbhd: List[List[int]] = []
    moves: List[Tuple[int, int]] = []
    n: int = len(soln) - 1

    local_costs: List[int] = []
    for i in range(n):
        curr: int = soln[i]
        nxt: int = soln[(i + 1)]
        cost: int = config.dms[str(n + 1)][curr][nxt]
        local_costs.append(cost)

    total_cost: int = sum(local_costs)
    probs: List[float] = [cost / total_cost for cost in local_costs]

    for _ in range(n_size):
        i_index: int = random.choices(range(n), weights=probs, k=1)[0]
        j_candidates: List[int] = [x for x in range(n) if x != i_index]
        j_index: int = random.choice(j_candidates)
        move: Tuple[int, int] = (
            min(soln[i_index], soln[j_index]),
            max(soln[i_index], soln[j_index]),
        )

        soln_mod: List[int] = soln.copy()
        soln_mod[i_index], soln_mod[j_index] = soln_mod[j_index], soln_mod[i_index]
        nbhd.append(soln_mod)
        moves.append(move)
        if len(nbhd) >= n_size:
            break
    return nbhd, moves
