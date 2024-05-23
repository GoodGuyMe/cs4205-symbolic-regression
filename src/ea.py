import time
from typing import Literal
from functools import cache

import numpy as np
import sympy as sym
import pandas as pd
import pygmo as pg

from src.utils import CSVLogger, debug_assert_fitness_correctness
from src.evaluation import get_fitness_and_parser
from src.initialisation import init_grow
from src.variation import get_variation_fn
from src.selection import select_single_objective, select_multi_objective

def DEPGEP(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    operators: list[str] = ("+", "-", "*", "/", "sin"),
    population_size: int = 25,
    max_expression_size: int = 32,
    num_constants: int = 5,
    scaling_factor: float = 0.2,
    p_crossover: float = 0.1,
    initialisation: str = "random",
    linear_scaling: bool = False,
    multi_objective: bool = False,
    max_generations: int | None = None,
    max_evaluations: int | None = None,
    max_time_seconds: float | None = None,
    seed: int | None = None,
    log_file: str | None = None,
    log_level: Literal["mse_elite", "non_dominated", "population"] = "non_dominated",
    log_frequency: int = 100,
    log_meta: dict | None = None,
    quiet: bool = False,
    return_value: Literal["mse_elite", "non_dominated"] | None = None,
    **kwargs
):
    """An implementation of DE-PGEP (https://doi.org/10.1145/1389095.1389331)."""

    assert X.shape[0] == y.shape[0]
    assert len(y.shape) == 1

    if not quiet:
        print("Compiling ... ", end="", flush=True)
    t_call = time.time()

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    test_data_available = X_test is not None
    if test_data_available:
        assert X.shape[1] == X_test.shape[1]
        assert X_test.shape[0] == y_test.shape[0]
        assert len(y_test.shape) == 1
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

    if log_meta is None:
        log_meta = {}
    log_meta["seed"] = seed if seed is not None else time.time_ns() % (2 ** 31 - 1)

    evaluate_individual, evaluate_population, to_sympy, perform_variation = get_compiled_functions(
        operators=operators,
        population_size=population_size,
        max_expression_size=max_expression_size,
        num_constants=num_constants,
        max_instances=max(X.shape[0], X_test.shape[0] if test_data_available else 0),
        num_inputs=X.shape[1],
        scaling_factor=scaling_factor,
        p_crossover=p_crossover,
        linear_scaling=linear_scaling
    )

    if multi_objective:
        perform_selection = select_multi_objective
    else:
        perform_selection = select_single_objective
    
    if not quiet:
        print(f"done after {time.time() - t_call:.3f}s")

    rng = np.random.Generator(np.random.Philox(seed=log_meta["seed"]))
    if log_file is None and not quiet:
        print(f"Using seed {log_meta['seed']}")

    # initialization
    structures = rng.random((population_size, max_expression_size), dtype=np.float32) * (len(operators) + X.shape[1] + num_constants)
    constants = rng.random((population_size, num_constants), dtype=np.float32)
    
    if initialisation == "random":
        pass # population is already initialized randomly
    elif initialisation == "grow":
        init_grow(structures, constants, operators, X.shape[1], rng)
    else:
        raise ValueError(f"Unknown initialisation: '{initialisation}'")

    fitness = np.empty((population_size, 2), dtype=np.float32)

    evaluate_population(structures, constants, fitness, X, y, linear_scaling)
    evaluations = population_size

    trial_structures = np.empty((population_size, max_expression_size), dtype=np.float32)
    trial_constants = np.empty((population_size, num_constants), dtype=np.float32)
    trial_fitness = np.empty((population_size, 2), dtype=np.float32)

    logger = CSVLogger(
        log_file,
        log_meta,
        log_level,
        X, y,
        X_test, y_test,
        linear_scaling,
        evaluate_individual,
        to_sympy
    )

    time_seconds = 0
    time_seconds_raw = 0
    generation = 0
    t_start = time.time()
    t_last_print = 0
    while (max_generations is None or generation < max_generations) \
        and (max_evaluations is None or evaluations < max_evaluations) \
        and (max_time_seconds is None or time_seconds < max_time_seconds):

        if generation % log_frequency == 0:
            logger.log(generation, evaluations, time_seconds, time_seconds_raw, structures, constants, fitness)

        generation_start = time.time()
        evaluations += perform_variation(structures, constants, fitness, trial_structures, trial_constants, trial_fitness, X, y, rng)
        perform_selection(structures, constants, fitness, trial_structures, trial_constants, trial_fitness)
        generation_end = time.time()

        # use this to check that the fitness matches the encoded expressions
        # debug_assert_fitness_correctness(structures, constants, fitness, to_sympy, X, y, linear_scaling)

        generation += 1
        time_seconds = generation_end - t_start
        time_seconds_raw += generation_end - generation_start

        if not quiet and generation % 10 == 0 and generation_end - t_last_print > 1.5:
            t_last_print = time.time()
            best_idx = fitness[:, 0].argmin()
            best_fitness = fitness[best_idx, 0]
            best_size    = fitness[best_idx, 1]
            print(f"Generation: {generation: 8d} | Evaluations: {int(evaluations): 10.2g} | Time [s]: {time_seconds: 7.2f} | Best fitness: [{best_fitness: 10.3g},{int(best_size): 3d}]")

    logger.log(generation, evaluations, time_seconds, time_seconds_raw, structures, constants, fitness)

    if not quiet:
        print(f"Achieved {evaluations / time_seconds:.2f}evaluations/second")

    if return_value == "mse_elite":
        best_idx = fitness[:, 0].argmin()
        best_fitness = fitness[best_idx]
        expression = to_sympy(structures[best_idx], constants[best_idx], X, y, linear_scaling, simplify=False, precision=3)
        if not quiet:
            print(f"{sym.simplify(expression)} @ (MSE: {best_fitness[0]}, Size: {best_fitness[1]})")
        return pd.DataFrame([dict(expression=expression, mse_train=best_fitness[0], size=int(best_fitness[1]))])
    elif return_value == "non_dominated":
        ndf_indices = pg.non_dominated_front_2d(fitness)
        _front = [dict(
            expression = to_sympy(structures[idx], constants[idx], X, y, linear_scaling, simplify=False, precision=3),
            mse_train = fitness[idx, 0],
            size = int(fitness[idx, 1])
        ) for idx in ndf_indices]
        # remove duplicates
        unique_solutions = set()
        front = []
        for solution in _front:
            if solution["expression"] not in unique_solutions:
                unique_solutions.add(solution["expression"])
                front.append(solution)
        if not quiet:
            best = min(front, key=lambda s: s["mse_train"])
            print(f"{sym.simplify(best['expression'])} @ (MSE: {best['mse_train']}, Size: {best['size']})")
        return pd.DataFrame(front)

@cache
def get_compiled_functions(
    operators: list[str],
    population_size: int,
    max_expression_size: int,
    num_constants: int,
    max_instances: int,
    num_inputs: int,
    scaling_factor: float,
    p_crossover: float,
    linear_scaling: bool
):
    """This function aims to avoid repeated jit compilations by caching"""
    evaluate_individual, evaluate_population, to_sympy = get_fitness_and_parser(
        max_expression_size=max_expression_size,
        num_constants=num_constants,
        max_instances=max_instances,
        num_inputs=num_inputs,
        operators=operators
    )

    perform_variation = get_variation_fn(
        population_size=population_size,
        max_expression_size=max_expression_size,
        num_constants=num_constants,
        library_size=len(operators) + num_constants + num_inputs,
        p_crossover=p_crossover,
        scaling_factor=scaling_factor,
        linear_scaling=linear_scaling,
        evaluate_individual=evaluate_individual,
        evaluate_population=evaluate_population
    )

    return (
        evaluate_individual,
        evaluate_population,
        to_sympy,
        perform_variation
    )

if __name__ == "__main__":
    # call as module, i.e. `python -m src.ea`
    from src.utils import synthetic_problem
    from sklearn.model_selection import train_test_split

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ground_truth = "0.3 * x0 * sin(2 * pi * x0)"
    X, y = synthetic_problem(ground_truth, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    so_front = DEPGEP(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        population_size=100,
        initialisation="grow",
        linear_scaling=False,
        multi_objective=False,
        max_time_seconds=30,
        return_value="non_dominated"
    )

    # with more objectives, larger population sizes and budgets are needed
    # - at least with the provided code, maybe you can improve on that...
    mo_front = DEPGEP(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        population_size=1000,
        linear_scaling=True,
        multi_objective=True,
        max_time_seconds=60,
        return_value="non_dominated"
    )

    so_front["type"] = "Single Objective"
    mo_front["type"] = "Multi Objective"
    fronts = pd.concat([so_front, mo_front], ignore_index=True)

    ax = sns.lineplot(
        fronts,
        x="size",
        y="mse_train",
        hue="type",
        marker="o",
        alpha=0.5,
        legend="brief"
    )
    for _, row in fronts.iterrows():
        ax.text(row["size"] + 0.2, row["mse_train"], row["expression"], fontsize=8)
    ax.set_yscale("log")
    ax.set_title(f"Pareto Approximation Fronts for {ground_truth}")
    plt.show()