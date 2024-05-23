import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.utils import get_problem_X_y, save_problem, load_problem
from src.ea import DEPGEP

# This file provides a possible starting point for the experiments you are going to perform

# Note that if your experiments includes a quantitative comparison, the setup should include
#  - multiple runs (>10)
#  - a computational budget permissive enough such that the methods tested are close to converging
#  - separate training and testing data are used with different seeds between runs

def run_once(train_path, test_path, **kwargs):
    try:
        X, y = load_problem(train_path)
        X_test, y_test = load_problem(test_path) if test_path is not None else (None, None)
        
        DEPGEP(X, y, X_test=X_test, y_test=y_test, **kwargs)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print("Run failed:", e)

def run_experiment(
    problems: list[str],
    methods: list[dict],
    folds: int = 5,
    repeats: int = 3,
    seed: int | None = 42,
    results_path: str = "results",
    clear_results_path: bool = False,
    max_workers: int = None
):
    if clear_results_path and os.path.exists(results_path):
        shutil.rmtree(results_path)
    if os.path.exists("data/tmp"):
        shutil.rmtree("data/tmp")
    os.makedirs(results_path, exist_ok=True)

    rng = np.random.Generator(np.random.Philox(seed=seed))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for pi, problem in enumerate(problems):
            X, y, synthetic = get_problem_X_y(problem)

            for fold, (train_indices, test_indices) in enumerate(
                KFold(n_splits=folds, shuffle=True, random_state=rng.integers(2 ** 31 - 1)).split(X)
            ):
                # data is written to a file to avoid dealing with shared memory
                train_path = f"data/tmp/p{pi}_f{fold}_train.tsv"
                save_problem(X, y, train_path, train_indices)
                test_path = f"data/tmp/p{pi}_f{fold}_test.tsv"
                save_problem(X, y, test_path, test_indices)

                for repeat in range(repeats):
                    run_seed = rng.integers(2 ** 31 - 1)

                    for mi, method in enumerate(methods):
                        log_file=f"{results_path}/p{pi}/m{mi}/f{fold}_r{repeat}.csv"

                        if not (os.path.exists(log_file) and os.path.isfile(log_file)):
                            futures.append(pool.submit(
                                run_once,
                                train_path,
                                test_path,
                                **method,
                                seed=run_seed,
                                log_file=log_file,
                                log_meta=dict(
                                    problem=problem,
                                    method=method.get("name", f"M{mi}"),
                                    fold=fold,
                                    repeat=repeat,
                                    synthetic=synthetic
                                )
                            ))

        progress = tqdm(total=len(futures))
        for f in as_completed(futures):
            e = f.exception()
            if e is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                raise e
            progress.update()


if __name__ == "__main__":
    logging_and_budget = dict(
        max_evaluations = int(5e5),
        max_time_seconds = int(30),
        log_level = "mse_elite",
        log_frequency = 100,
        # quiet = True
    )
    
    run_experiment(
        problems=[
            # from https://doi.org/10.1145/1389095.1389331
            "2.718 * x0 ** 2 + 3.141636 * x0",
            "x0 ** 3 - 0.3 * x0 ** 2 - 0.4 * x0 - 0.6",
            "0.3 * x0 * sin(2 * pi * x0)",
            # from https://archive.ics.uci.edu/datasets
            "Airfoil",
            "Concrete Compressive Strength",
            "Energy Cooling",
            "Energy Heating",
            "Yacht Hydrodynamics",
        ],
        methods=[
            dict(
                name="Fewer Operators",
                operators=tuple("+,-,*,/,sin".split(",")),
                max_expression_size=32,
                num_constants=5,
                population_size=100,
                **logging_and_budget
            ),
            dict(
                name="More Operators",
                operators=tuple("+,-,*,/,sin,cos,exp,log,sqrt".split(",")),
                max_expression_size=32,
                num_constants=5,
                population_size=100,
                **logging_and_budget
            )
        ],
        folds=5,
        repeats=3,
        # clear_results_path=True
    )