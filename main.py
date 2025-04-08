import csv
import sys
import time
import os
from statistics import mean
from datetime import datetime
from typing import Any, List, Tuple
import numpy as np

import config
from algorithms.current import core1, core2, core3
from algorithms.enhancements import search

final_output: str = ""
current_timestamp: str = datetime.now().strftime("%m%d_%H%M%S")
config.iter_max: int = 100
test = []


def run_core_simulation(core_module: Any, core_name: str) -> None:
    global final_output
    output: str = ""

    for n in range(len(config.pois)):
        for j in range(len(config.tenures)):
            list_soln_best: List[float] = []
            list_soln_best_tracker: List[List[float]] = []
            list_time: List[float] = []

            for i in range(config.runs):
                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    f"\r(STATUS: {core_name}) POI: {n + 1}/{len(config.pois)} | Tenure: {j + 1}/{len(config.tenures)} | run: {i + 1}/{config.runs}"
                )
                sys.stdout.flush()

                s_time: float = time.perf_counter()
                soln_best, soln_best_tracker = core_module.search(
                    soln_init=config.soln_inits[config.pois[n]],
                    tabu_tenure=config.tenures[j],
                    iter_max=config.iter_max,
                )
                e_time: float = time.perf_counter()

                list_soln_best.append(soln_best)
                list_soln_best_tracker.append(soln_best_tracker)
                list_time.append(e_time - s_time)

            soln_lst: List[float] = [
                soln for run in list_soln_best_tracker for soln in run
            ]

            avg_soln: float = round(mean(soln_lst), 2)
            avg_time: float = round(mean(list_time), 2)
            data = np.array(soln_lst)
            cv = round(np.std(data) / np.mean(data), 2)

            output += f"POI: {config.pois[n]} | Tenure: {config.tenures[j]}\n"
            output += f"avg soln: {avg_soln}\n"
            output += f"avg time: {avg_time}\n"
            output += f"DS (CV): {cv}\n"
            output += "================\n\n"

            test.append([
                core_name,
                config.pois[n],
                list_time,
                soln_lst,
                config.tenures[j],
            ])

    print()
    print("\nFinished\n\n")
    print(output)

    final_output += f"\n=====Result ({core_name})=====\n" + output

    with open(f"{core_name}.csv", "w", newline="") as file:
        writer = csv.writer(file)

        for item in test:
            writer.writerow(item)


def run_hybrid_simulation() -> None:
    global final_output
    output: str = ""

    for n in range(len(config.pois)):
        list_soln_best: List[float] = []
        list_soln_best_tracker: List[List[float]] = []
        list_time: List[float] = []

        for i in range(config.runs):
            sys.stdout.write("\r\033[K")
            sys.stdout.write(
                f"\r(STATUS: Hybrid1) POI: {config.pois[n]} | run: {i + 1}/{config.runs}"
            )
            sys.stdout.flush()

            s_time: float = time.perf_counter()
            soln_best, soln_best_tracker = search(
                soln_init=config.soln_inits[config.pois[n]],
                iter_max=config.iter_max,
            )
            e_time: float = time.perf_counter()

            list_soln_best.append(soln_best)
            list_soln_best_tracker.append(soln_best_tracker)
            list_time.append(e_time - s_time)

        soln_lst: List[float] = [soln for run in list_soln_best_tracker for soln in run]

        avg_soln: float = round(mean(soln_lst), 2)
        avg_time: float = round(mean(list_time), 2)
        data = np.array(soln_lst)
        cv = round(np.std(data) / np.mean(data), 2)

        output += f"POI: {config.pois[n]}\n"
        output += f"avg soln: {avg_soln}\n"
        output += f"avg time: {avg_time}\n"
        output += f"DS (CV): {cv}\n"
        output += "================\n\n"

        test.append(
            [
                "enhanced",
                config.pois[n],
                list_time,
                soln_lst,
            ]
        )

    print()
    print("\nFinished\n\n")
    print(output)

    final_output += "\n=====Result (Enhanced)=====\n" + output


if __name__ == "__main__":
    run_core_simulation(core1, "core1")
    run_core_simulation(core2, "core2")
    run_core_simulation(core3, "core3")
    run_hybrid_simulation()

    output_dir: str = os.path.join(os.path.dirname(__file__), "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file_name: str = f"result_{current_timestamp}.txt"
    output_file_path: str = os.path.join(output_dir, output_file_name)

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(final_output)

    with open("output-soln.csv", "w", newline="") as file:
        writer = csv.writer(file)

        # Create header row with Time(POI), Solution(POI) for each POI
        header = []
        for poi in range(len(test)):
            if len(test[poi]) > 4:
                header.extend([f"{test[poi][0]} - POI: {test[poi][1]} - Tenure: {test[poi][4]}", "Time"])
            else:
                header.extend([f"{test[poi][0]} - POI: {test[poi][1]}", "Time"])
        writer.writerow(header)

        # Determine the maximum number of data rows needed
        max_rows = 0
        for poi in range(len(test)):
            rows_needed = max(len(test[poi][2]), len(test[poi][3]))
            max_rows = max(max_rows, rows_needed)

        # Write data rows
        for row in range(max_rows):
            row_data = []
            for poi in range(len(test)):
                if row < len(test[poi][3]):
                    row_data.append(test[poi][3][row])
                else:
                    row_data.append("")  # Empty cell if no more solutions

                # Add time (if available for this row)
                if row < len(test[poi][2]):
                    row_data.append(test[poi][2][row])
                else:
                    row_data.append("")

            writer.writerow(row_data)
