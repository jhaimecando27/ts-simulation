import random
import math
from typing import List, Tuple

import config
from algorithms.utils import val, best_admissible_soln


def search(
    soln_init: List[int],
    iter_max: int = 100,
) -> Tuple[int, List[int], int]:
    tabu_list: List[Tuple[int, int]] = []
    soln_curr: List[int] = soln_init.copy()
    soln_best: List[int] = soln_init.copy()
    soln_best_tracker: List[int] = []

    stagnant_ctr: int = 0
    stagnant_best: int = 0
    improve_ctr: int = 0

    tabu_tenure: int = math.floor(len(soln_init) * 0.1)
    adjustment_rate: float = 0.0
    solution_diversity_tracker: List[List[int]] = []

    for iter_ctr in range(iter_max):

        if stagnant_ctr or improve_ctr > 10:
            solution_diversity_tracker.append(soln_curr.copy())
            unique = len(set(map(tuple, solution_diversity_tracker)))
            total = max(1, len(solution_diversity_tracker))
            solution_diversity = unique / total  # 0.0 to 1.0

            adjustment_rate = max(1, improve_ctr) / max(1, stagnant_ctr)

            tabu_tenure = wave_tenure_adaptation(
                soln_init=soln_init,
                base_tenure=tabu_tenure,
                iter_ctr=iter_ctr,
                iter_max=iter_max,
                solution_diversity=solution_diversity,
                adjustment_rate=adjustment_rate,
            )

        if stagnant_ctr:
            soln_curr = wave_resonance_perturbation(
                soln_curr, iter_ctr, iter_max, soln_best, stagnant_ctr
            )

        nbhd, moves = neighborhood(soln_curr, tabu_list)
        nbhr_best, move_best = best_admissible_soln(nbhd, moves, tabu_list, soln_best)

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best.copy()

            if stagnant_ctr > stagnant_best:
                stagnant_best = stagnant_ctr

            stagnant_ctr = 0
            improve_ctr += 1
        else:
            stagnant_ctr += 1
            improve_ctr = 0

        soln_best_tracker.append(val(nbhr_best))

        soln_curr = nbhr_best.copy()
        tabu_list.append(move_best)
        while len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker


def neighborhood(
    soln: List[int],
    tabu_list: List[Tuple[int, int]],
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    nbhd: List[List[int]] = []
    moves: List[Tuple[int, int]] = []
    n: int = len(soln)

    segment_costs: List[Tuple[int, float]] = []
    for i in range(n):
        next_idx: int = (i + 1) % n
        cost: float = config.dms[str(len(soln))][soln[i]][soln[next_idx]]
        segment_costs.append((i, cost))

    segment_costs.sort(key=lambda x: x[1], reverse=True)

    k: int = int(n**0.3)
    focal_indices: List[int] = [p[0] for p in segment_costs[:k]]

    non_focal: List[int] = [i for i in range(n) if i not in focal_indices]
    if non_focal:
        random_idx: int = random.choice(non_focal)
        if random_idx not in focal_indices:
            focal_indices.append(random_idx)

    for i in focal_indices:
        radius: int = int(2 * n)

        j_candidates: set[int] = set()

        for offset in range(1, min(radius + 1, n)):
            j_candidates.add((i + offset) % n)
            j_candidates.add((i - offset + n) % n)

        for focal in focal_indices:
            if focal != i:
                j_candidates.add(focal)

        for j in [j for j in j_candidates if j > i and j < n]:
            soln_mod: List[int] = soln.copy()
            soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
            move: Tuple[int, int] = (soln[i], soln[j])

            if move in tabu_list:
                continue

            nbhd.append(soln_mod)
            moves.append(move)

    return nbhd, moves


def wave_resonance_perturbation(
    soln_curr: List[int],
    iter_ctr: int,
    iter_max: int,
    soln_best: List[int],
    stagnant_ctr: int,
) -> List[int]:
    n: int = len(soln_curr)

    progress_metrics: dict[str, float] = {
        "iter_progress": iter_ctr / iter_max,
        "stagnation_factor": min(1, stagnant_ctr / iter_max),
        "scale_factor": 1 + (n // 10) * 0.5,
    }

    perturbation_intensity: float = (
        progress_metrics["iter_progress"] + progress_metrics["stagnation_factor"]
    ) * progress_metrics["scale_factor"]

    wave_amplitude: int = max(
        1,
        int(
            n
            * (1 - perturbation_intensity)
            * (1 + stagnant_ctr / iter_ctr)
            * (1 + math.log(n) / 10)
        ),
    )

    resonance_factor: float = math.sin(
        perturbation_intensity * math.pi * 2 * (1 + math.log(n) / 10)
    )

    perturbed_soln: List[int] = soln_curr.copy()

    for _ in range(wave_amplitude):
        wave_centers: List[int] = [
            int(
                n
                * abs(
                    math.sin(
                        i
                        * resonance_factor
                        * (1 + stagnant_ctr / iter_max)
                        * (1 + math.log(n) / 10)
                    )
                )
            )
            for i in range(wave_amplitude)
        ]

        for center in wave_centers:
            wave_radius: int = max(
                1,
                int(
                    wave_amplitude
                    * (1 - abs(resonance_factor))
                    * (1 + stagnant_ctr / iter_max)
                    * (1 + math.log(n) / 10)
                ),
            )

            swap_candidates: set[int] = set()
            for offset in range(-wave_radius, wave_radius + 1):
                candidate: int = (center + offset) % n
                swap_candidates.add(candidate)

            # Unique swap strategy
            if len(swap_candidates) > 1:
                swap_point1, swap_point2 = random.sample(list(swap_candidates), 2)

                # Swap with probability based on solution quality
                swap_probability = max(0.3, 1 - val(perturbed_soln) / val(soln_best))

                if random.random() < swap_probability:
                    if val(perturbed_soln) < val(soln_best):
                        perturbed_soln[swap_point1], perturbed_soln[swap_point2] = (
                            perturbed_soln[swap_point2],
                            perturbed_soln[swap_point1],
                        )

    return perturbed_soln

def wave_tenure_adaptation(
    soln_init: list[int],
    base_tenure: int,
    iter_ctr: int,
    iter_max: int,
    solution_diversity: float,
    adjustment_rate: float,
) -> int:
    n = len(soln_init)
    wave = math.sin(2 * math.pi * iter_ctr / n)

    stagnation_penalty = 1 / (n + adjustment_rate)

    tenure_scale = (1 - stagnation_penalty * abs(wave)) * solution_diversity
    adjusted_tenure = base_tenure * tenure_scale

    return max(1, int(adjusted_tenure))
