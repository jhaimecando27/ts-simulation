import random
from typing import List, Tuple

import config
from algorithms.utils import neighborhood, val, best_admissible_soln


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
        nbhd_neigh, moves_neigh = neighborhood(soln_curr, tabu_list)
        nbhd_cross, moves_cross = crossover(nbhd_neigh, moves_neigh, tabu_list)
        nbhd_mut, moves_mut = mutation(nbhd_neigh, moves_neigh, tabu_list)

        combined_nbhd: List[List[int]] = nbhd_neigh + nbhd_cross + nbhd_mut
        combined_moves: List[Tuple[int, int]] = moves_neigh + moves_cross + moves_mut

        nbhr_best, move_best = best_admissible_soln(
            combined_nbhd, combined_moves, tabu_list, soln_best
        )

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best.copy()
            soln_best_tracker.append(val(soln_best))
        soln_curr = nbhr_best.copy()

        tabu_list.append(move_best)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker


def crossover(
    nbhd: List[List[int]],
    moves: List[Tuple[int, int]],
    tabu_list: List[Tuple[int, int]],
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    new_nbhd: List[List[int]] = []
    new_moves: List[Tuple[int, int]] = []
    parent1: List[int] = nbhd[0]
    parent2: List[int] = nbhd[1]
    n: int = len(parent1)
    cross_point: int = min(1, int(n**0.9))
    child: List[int] = parent1[:cross_point] + [
        gene for gene in parent2 if gene not in parent1[:cross_point]
    ]
    move: Tuple[int, ...] = (*moves[0], *moves[1])
    simplified_move: Tuple[int, int] = (move[0], move[-1])
    if simplified_move not in tabu_list:
        new_nbhd.append(child)
        new_moves.append(simplified_move)
    return new_nbhd, new_moves


def mutation(
    nbhd: List[List[int]],
    moves: List[Tuple[int, int]],
    tabu_list: List[Tuple[int, int]],
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    new_nbhd: List[List[int]] = []
    new_moves: List[Tuple[int, int]] = []
    for soln, move in zip(nbhd, moves):
        n: int = min(1, int(len(soln) ** 0.05))
        for i in range(n):
            for j in range(i + 1, n):
                soln_mod: List[int] = soln.copy()
                soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
                new_move: Tuple[int, int] = (soln_mod[i], soln_mod[j])
                if new_move not in tabu_list:
                    new_nbhd.append(soln_mod)
                    new_moves.append(new_move)
    return new_nbhd, new_moves
