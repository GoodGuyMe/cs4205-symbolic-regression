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



#### ea.py ###

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

### evaluation.py ###

import os
import sys
import importlib
import pathlib
import hashlib

import numpy as np
import numba as nb
from numba import types as nty
import sympy as sym

from src.utils import get_arity, OPERATORS

CACHE_DIR = pathlib.Path(".numba")

def get_fitness_and_parser(
        max_expression_size: int,
        num_constants: int,
        max_instances: int,
        num_inputs: int,
        operators: list[str]
):
    """Returns evaluation functions and expression -> sympy parsing function.
    
    Uses python metaprogramming to create a jit compiled evaluation function that can be cached.
    - Code for resolving operators is dynamically hardcoded to be able to use numba nopython mode
    - The code is then stored in a python file to make numba caching work and dynamically imported
    """
    assert len(set(operators)) == len(operators), "Duplicate operators"

    op_ids = [op[0] for op in OPERATORS]
    op_indices = [op_ids.index(op) for op in operators]
    num_operators = np.int32(len(op_indices))

    # 1. lookup tables
    fmt = [OPERATORS[op][1] for op in op_indices]
    arity = np.array(list(map(get_arity, fmt)) \
        + [0 for _ in range(num_inputs + num_constants)]).astype(np.int32)
    
    # 2. peak python metaprogramming
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Windows apparently does not like characters like * or + in file paths, so we use a hash...
    key = hashlib.sha256(f"{max_expression_size}_{num_constants}_{max_instances}_{num_inputs}_{'_'.join(operators)}".encode("ascii")).hexdigest()
    path = (CACHE_DIR / f"{key}.py")

    preamble = "\n".join([
        "import numpy as np",
        "import numba as nb",
        "from numba import types as nty",
        "",
        f"max_expression_size = {max_expression_size}",
        f"num_constants = {num_constants}",
        f"max_instances = {max_instances}",
        f"num_inputs = {num_inputs}",
        f"num_operators = {num_operators}",
        f"arity = np.array([{', '.join(map(str, list(arity)))}]).astype(np.int32)",
        ""
    ])
    operator_table = "                if " + "\n                elif ".join([
        f"op == {op_value}:" + "\n                    eval_buffer[:X.shape[0], buffer_idx] = np." \
            + OPERATORS[op_idx][2].__name__ + "(" + ", ".join([
                f"eval_buffer[:X.shape[0], eval_stack[arg_stack_size{f' + {i}' if i > 0 else ''}, 3]]" \
                for i in range(arity[op_value])
            ]) + ")" for op_value, op_idx in enumerate(op_indices)
        ])
    fstr = f"""
@nb.jit((
    nty.Array(nty.float32, 1, "C", aligned=True),
    nty.Array(nty.float32, 1, "C", aligned=True),
    nty.Array(nty.float32, 2, "C", aligned=True),
    nty.Array(nty.float32, 2, "C", aligned=True, readonly=True)
), nopython=True, nogil=True, fastmath={{"nsz", "arcp", "contract", "afn"}}, error_model="numpy", parallel=False, inline="always", cache=True)
def compute_output(structure, constants, eval_buffer, X):
    "Computes the expresssion output into the evaluation buffer and returns the expression size."
    # this function iterates over the structure from left to right, pushing functions where the arguments
    # still need to be evaluated on a stack and evaluating from the stack when possible, until finally returning
    # the output corresponding to the first node or failing if the expression is not valid

    # two stacks, one for keeping track of the operations that need to be evaluated, and one for the arguments
    eval_stack = np.empty(shape=(max_expression_size, 4), dtype=np.int32)

    op_stack_size  = np.int32(0) # this stack contains buffer index, operator index, remaining arity
    arg_stack_size = np.int32(0) # this stack contains argument output indices
    j = np.int32(0)              # current node index
    while (j == 0 or op_stack_size > 0) and j < max_expression_size:
        if op_stack_size > 0 and eval_stack[op_stack_size - 1, 2] <= 0:
            # Is the stack non-empty and we can compute something?
            op_stack_size -= 1
            buffer_idx = eval_stack[op_stack_size, 0]
            op = eval_stack[op_stack_size, 1]
            
            # update arity and arguments left for the parent node
            arg_stack_size -= arity[op]
            if op_stack_size > 0:
                eval_stack[op_stack_size - 1, 2] -= 1
            
            if op < num_operators:
                # the order of the arguments on the stack is inverted, but we already
                # decreased the stack size, so they are in the correct order again...
                # eval_buffer[:, buffer_idx] = ops[op](*[eval_buffer[:, arg_stack_size + ai] for ai in range(arity[op])])
{operator_table}
                # The operator table will look like the following, with all operators used:
                # if op == 0:
                #     eval_buffer[:X.shape[0], buffer_idx] = eval_buffer[:X.shape[0], eval_stack[arg_stack_size, 3]] + eval_buffer[:X.shape[0], eval_stack[arg_stack_size + 1, 3]]
                # elif op == 1:
                #     eval_buffer[:X.shape[0], buffer_idx] = eval_buffer[:X.shape[0], eval_stack[arg_stack_size, 3]] - eval_buffer[:X.shape[0], eval_stack[arg_stack_size + 1, 3]]
            else:
                op -= num_operators
                if op < X.shape[1]:
                    eval_buffer[:X.shape[0], buffer_idx] = X[:, op]
                else:
                    eval_buffer[:X.shape[0], buffer_idx] = constants[op - X.shape[1]]
        else: # if not, we get the next argument
            op = int(abs(structure[j]))

            eval_stack[op_stack_size, 0] = j         # index for evaluation buffer
            eval_stack[op_stack_size, 1] = op        # operator index
            eval_stack[op_stack_size, 2] = arity[op] # number of arguments left
            op_stack_size += 1

            eval_stack[arg_stack_size, 3] = j
            arg_stack_size += 1

            j += 1
    return j

@nb.jit((
    nty.Array(nty.float32, 1, "C", aligned=True),
    nty.Array(nty.float32, 1, "C", aligned=True),
    nty.Array(nty.float32, 1, "C", aligned=True),
    nty.Array(nty.float32, 2, "C", aligned=True, readonly=True),
    nty.Array(nty.float32, 1, "C", aligned=True, readonly=True),
    nty.boolean
), nopython=True, nogil=True, fastmath={{"nsz", "arcp", "contract", "afn"}}, error_model="numpy", cache=True, parallel=False)
def evaluate_individual(structure, constants, fitness, X, y, linear_scaling):
    eval_buffer = np.empty(shape=(max_instances, max_expression_size), dtype=np.float32)
    expression_size = compute_output(structure, constants, eval_buffer, X)

    if expression_size < max_expression_size:
        if np.isfinite(eval_buffer[:X.shape[0], 0]).all():
            if linear_scaling:
                eval_buffer[:X.shape[0], 1] = 1
                w, b = np.linalg.lstsq(eval_buffer[:X.shape[0], :2], y)[0]
                eval_buffer[:X.shape[0], 0] = w * eval_buffer[:X.shape[0], 0] + b
            fitness[0] = np.mean((eval_buffer[:X.shape[0], 0] - y) ** 2)
        else:
            fitness[0] = np.inf
        fitness[1] = expression_size
    else:
        fitness[:] = np.inf

@nb.jit((
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C", readonly=True),
    nty.Array(nty.float32, 1, "C", readonly=True),
    nty.boolean
    ), nopython=True, nogil=True, fastmath={{"nsz", "arcp", "contract", "afn"}}, error_model="numpy", cache=True, parallel=False
)
def evaluate_population(structures, constants, fitness, X, y, linear_scaling):
    for i in range(structures.shape[0]):
        evaluate_individual(structures[i], constants[i], fitness[i], X, y, linear_scaling)
"""

    code = preamble + fstr

    # 3. cache invalidation
    overwrite = True
    if path.exists():
        with open(path, "rb") as f:
            overwrite = f.read() != code.encode("utf-8")

    if overwrite:
        with open(path, "+w", encoding="utf-8") as f:
            f.write(code)
    
    # 4. importing the code
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)

    compute_output = module.compute_output
    evaluate_individual = module.evaluate_individual
    evaluate_population = module.evaluate_population

    # 5. representation -> sympy
    sym_buffer = ["" for _ in range(max_expression_size)]
    def to_sympy(structure: np.ndarray, constants: np.ndarray, X: np.ndarray, y: np.ndarray, linear_scaling: bool = False, simplify: bool = False, precision: int | None = None) -> str | None:
        """Returns a `sympy` compatible model of the encoded expression if it is valid, else `None`.

        Parameters:
        ----------
        structure: np.ndarray
            The structure to parse
        constants: np.ndarray
            The constants to parse
        X: np.ndarray
            The training (!) data
        y: np.ndarray
            The training (!) targets
        linear_scaling: bool
            If enabled, the linear scaling coefficients are computed and added to the expression
        simplify: bool
            If `True`, `sympy` is used to simplify the model
        precision: int | None
            If set, constant values are truncated to the requested precision
        """
        assert precision is None or precision > 0

        op_stack = []
        arg_stack = []

        j = 0
        while (j == 0 or len(op_stack) > 0) and j < max_expression_size:
            if len(op_stack) > 0 and op_stack[-1][2] <= 0:
                buf_idx, op, _ = op_stack.pop()
                _arity = arity[op]
                if op < num_operators:
                    sym_buffer[buf_idx] = fmt[op] \
                        .format(*[sym_buffer[arg_stack[ai - _arity]] for ai in range(_arity)])
                else:
                    op -= num_operators

                    if op < num_inputs:
                        sym_buffer[buf_idx] = f"x{op}"
                    else:
                        value = constants[op - num_inputs]
                        sym_buffer[buf_idx] = str(value) if precision is None or simplify else f"{value:.{precision}g}"
                arg_stack = arg_stack[:len(arg_stack) - _arity]
                if len(op_stack) > 0:
                    op_stack[-1][2] -= 1
            else:
                op = int(abs(structure[j]))
                op_stack.append([j, op, arity[op]])
                arg_stack.append(j)
                j += 1
        if j >= max_expression_size:
            return None
        
        if linear_scaling:
            eval_buffer = np.empty(shape=(max_instances, max_expression_size), dtype=np.float32)
            compute_output(structure, constants, eval_buffer, X)
            eval_buffer[:X.shape[0], 1] = 1
            w, b = np.linalg.lstsq(eval_buffer[:X.shape[0], :2], y, rcond=None)[0]
            sym_buffer[0] = f"{b} + {w} * ({sym_buffer[0]})"

        if not simplify:
            return sym_buffer[0]

        e = sym.simplify(sym.sympify(sym_buffer[0]), ratio=1.0)
        if precision is None:
            return str(e)

        for n in sym.preorder_traversal(e):
            if isinstance(n, sym.Float):
                e = e.subs(n, sym.Float(n, precision + 1))
        return e

    return evaluate_individual, evaluate_population, to_sympy

### initialisation.py ###


from src.utils import get_arity, OPERATORS

def init_grow(structures, constants, operators, num_inputs, rng):
    """Basic grow initialisation for the prefix notation representation."""

    op_indices = [op_id for op_id,_,_ in OPERATORS]
    
    def grow_prefix():
        """Grow initialisation, but with a prefix notation representation"""
        if rng.random() < 0.5: # 50% chance of getting a terminal
            if rng.random() < 0.5: # 50% chance of getting a input feature/constant
                return [len(operators) + rng.integers(num_inputs)]
            else:
                return [len(operators) + num_inputs + rng.integers(constants.shape[1])]
        else:
            idx = rng.choice(len(operators))
            op_idx = op_indices.index(operators[idx])
            arity = get_arity(OPERATORS[op_idx][1])
            return [idx] + [node for _ in range(arity) for node in grow_prefix()]

    for i in range(structures.shape[0]):
        tree = grow_prefix()
        # ! trees can be longer than allowed...
        l = min(structures.shape[1], len(tree))
        structures[i, :l] = tree[:l]
        # add some more randomness in [0, 1) to increase diversity
        structures[i, :l] += rng.random((l,), dtype=np.float32)

    # init constants in the range [-10, 10) instead of [0, 1)
    constants = rng.random(constants.shape, dtype=np.float32) * 20 - 10

### selection.py ###

@nb.jit((
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C")
), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
def select_single_objective(
    structures,
    constants,
    fitness,
    trial_structures,
    trial_constants,
    trial_fitness
):
    for i in range(structures.shape[0]):
        # replace solutions dominated by the trial solution
        if trial_fitness[i, 0] <= fitness[i, 0]:
            structures[i,:] = trial_structures[i,:]
            constants[i,:]  = trial_constants[i,:]
            fitness[i, :]   = trial_fitness[i, :]

@nb.jit((
    nty.Array(nty.float32, 1, "C"),
    nty.Array(nty.float32, 1, "C"),
), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
def dominates(fitness1, fitness2):
    strictly_better_somewhere = False
    for i in range(fitness1.shape[0]):
        f1_ok, f2_ok = np.isfinite(fitness1[i]), np.isfinite(fitness2[i])
        both_ok = f1_ok and f2_ok
        if (not f1_ok and f2_ok) or (both_ok and fitness1[i] > fitness2[i]):
            return False
        elif (f1_ok and not f2_ok) or (both_ok and fitness1[i] < fitness2[i]):
            strictly_better_somewhere = True
    return strictly_better_somewhere

@nb.jit((
    nty.Array(nty.float32, 2, "C"),
    nty.int64,
), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
def fast_non_dominated_sorting(fitness, target_size):
    """Non-dominated sorting as per https://doi.org/10.1109/4235.996017
    
    If target size is less than the number of fitness instances, then the method might stop early and thus only ranks present in a front are accurate.
    """
    size = fitness.shape[0]
    dominated_by = [nb.typed.List.empty_list(nty.int32) for _ in range(size)]
    domination_count = [0 for _ in range(size)]
    fronts = nb.typed.List()
    fronts.append(nb.typed.List.empty_list(nty.int32))
    ranks = [np.int32(0) for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if dominates(fitness[i], fitness[j]):
                dominated_by[i].append(np.int32(j))
            elif dominates(fitness[j], fitness[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            ranks[i] = 0
            fronts[0].append(np.int32(i))
    
    total = 0
    while len(fronts[-1]) > 0:
        total += len(fronts[-1])
        if total >= target_size:
            return ranks, fronts
        
        fronts.append(nb.typed.List.empty_list(nty.int32))
        for i in fronts[-2]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    ranks[j] = len(fronts) - 1
                    fronts[-1].append(np.int32(j))
    
    return ranks, fronts[:-1]

@nb.jit((
    nty.Array(nty.float32, 2, "C"),
), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
def crowding_distance(fitness):
    """Crowding distance as per https://doi.org/10.1109/4235.996017"""
    size, num_objectives = fitness.shape

    distance = np.zeros((size,), dtype=np.float32)
    for j in range(num_objectives):
        indices = np.argsort(fitness[:, j])
        distance[indices[0]] = distance[indices[-1]] = np.inf

        k = size - 1
        while k > 0 and not np.isfinite(fitness[indices[k], j]):
            k -= 1
        objective_range = fitness[indices[k], j] - fitness[indices[0], j]

        for i in range(1, k - 1):
            distance[indices[i]] += (fitness[indices[i+1], j] - fitness[indices[i-1], j]) \
                    / objective_range
    
    return distance

@nb.jit(nopython=True, nogil=True, cache=True, parallel=False, inline="always")
def numpy_index(arr, indices):
    res = np.empty((len(indices), arr.shape[1]), dtype=arr.dtype)
    for j, i in enumerate(indices):
        res[j, :] = arr[i, :]
    return res

@nb.jit((
    nty.Array(nty.float32, 2, "C"),
    nty.int64,
), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
def nsgaII_selection(fitness, target_size):
    """Multi-objective selection as per https://doi.org/10.1109/4235.996017"""
    _ranks, fronts = fast_non_dominated_sorting(fitness, target_size)

    indices = nb.typed.List.empty_list(nty.int64)
    i = 0
    while i < len(fronts) and len(indices) + len(fronts[i]) <= target_size:
        for j in fronts[i]:
            indices.append(j)
        i += 1
    
    if i < len(fronts) and len(indices) < target_size:
        by_distance = np.argsort(crowding_distance(numpy_index(fitness, fronts[i])))
        for j in range(target_size - len(indices)):
            indices.append(fronts[i][by_distance[len(by_distance) - 1 - j]])

    return indices

@nb.jit((
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C"),
    nty.Array(nty.float32, 2, "C")
), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
def select_multi_objective(
    structures,
    constants,
    fitness,
    trial_structures,
    trial_constants,
    trial_fitness
):
    population_size = structures.shape[0]
    joint_fitness = np.empty((2 * population_size, 2), dtype=np.float32)
    joint_fitness[:population_size, :] = fitness
    joint_fitness[population_size:, :] = trial_fitness

    indices = sorted(nsgaII_selection(joint_fitness, population_size))
    
    # since the surviving indices are sorted, we can just replace the population indices that
    # do not make it (i.e. where i < indices[start]) with surviving indices from the trial population
    start, end = 0, population_size
    for i in range(population_size):
        if i < indices[start]:
            end -= 1
            idx = indices[end] - structures.shape[0]
            structures[i, :] = trial_structures[idx, :]
            constants[i, :]  = trial_constants[idx, :]
            fitness[i, :]    = trial_fitness[idx, :]
        else:
            start += 1

### utils.py ###
import numpy as np
import sympy as sym
import pygmo as pg

OPERATORS = [
    # Add a tuple consisting of (op_id, format string, vectorized function)
    # for each operator you want to use (arity is derived from the format string)
    # Note: currently, only numpy functions are supported!
    ("+", "({} + {})", np.add),
    ("-", "({} - {})", np.subtract),
    ("*", "({} * {})", np.multiply),
    ("/", "({} / {})", np.divide),
    ("sin", "sin({})", np.sin),
    ("cos", "cos({})", np.cos),
    ("exp", "exp({})", np.exp),
    ("log", "log({})", np.log),
    ("sqrt", "sqrt({})", np.sqrt)
]

def get_arity(fmt: str) -> int:
    """Gets the number of arguments from a format string."""
    return len([1 for _,a,_,_ in string.Formatter().parse(fmt) if a != None])

def get_problem_X_y(problem: str, **kwargs):
    """Returns the X,y pair of input and output values for the available problems."""
    datasets = [
        ("Airfoil", "data/airfoil_full.tsv"),
        ("Concrete Compressive Strength", "data/concrete_full.tsv"),
        ("Energy Cooling", "data/energycooling_full.tsv"),
        ("Energy Heating", "data/energyheating_full.tsv"),
        ("Yacht Hydrodynamics", "data/yacht_full.tsv")
    ]
    matches = [ppath for pname, ppath in datasets if problem == pname]
    if len(matches) > 0:
        data = np.loadtxt(matches[0], delimiter=" ")
        return data[:,:-1], data[:,-1], False
    else:
        return *synthetic_problem(problem, **kwargs), True

def save_problem(X, y, filename, indices=None):
    if indices is not None:
        X = X[indices, :]
        y = y[indices]
    pdir = os.path.dirname(filename)
    if len(pdir) > 0:
        os.makedirs(pdir, exist_ok=True)
    np.savetxt(filename, np.hstack([X, y.reshape(-1, 1)]), fmt="%+.17g", delimiter=" ", encoding="ascii")

def load_problem(filename):
    Xy = np.loadtxt(filename, delimiter=" ", encoding="ascii")
    return Xy[:, :-1], Xy[:, -1]

def lambdify_expression(e: str | sym.Expr):
    """Converts a `sympy` compatible expression string into a function accepting a dataset `X`."""
    e = str(e)

    symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:,int(s[1:])] for s in symbols.keys()])
        except Exception as e:
            print(e)
            return np.repeat(float("nan"), X.shape[0])
    return fn

def synthetic_problem(expr: str, size: int = 1000, lb: float = -10.0, ub: float = 10.0, noise: float = 0.01, random_state: int | None = None):
    """Creates a synthetic problem by sampling a random dataset, applying the function and possibly adding noise."""
    assert ub > lb, "Invalid initialisation bounds"

    rng = np.random.Generator(np.random.Philox(seed=random_state))

    num_inputs = max([int(x) + 1 for x in re.findall(r"x(\d+)", expr)])

    X = rng.random(size=(size, num_inputs)) * (ub - lb) + lb
    y = lambdify_expression(expr)(X)
    if noise > 0:
        y += rng.standard_normal(size) * noise
    return X, y

def debug_assert_fitness_correctness(structures, constants, fitness, to_sympy, X, y, linear_scaling):
    """Asserts that the fitness computed matches the fitness of the corresponding expression."""
    for i in range(structures.shape[0]):
        if np.isfinite(fitness[1]).all():
            e = to_sympy(structures[i], constants[i], X, y, linear_scaling)
            if e is not None:
                try:
                    f = lambdify_expression(e)
                except Exception:
                    print(f"Could not lambdify '{e}' (MSE: {fitness[0]}, Size: {fitness[1]})")
                    continue
                y_pred = f(X)
                mse = np.mean((y - y_pred) ** 2)
                is_ok = np.allclose(fitness[i, 0], mse, rtol=0.01, atol=1e-4)
                assert is_ok, f"Got MSE of {fitness[i, 0]}, but expected {mse}"

class CSVLogger:
    def __init__(
        self,
        log_file: str,
        log_meta: dict,
        log_level: Literal["mse_elite", "non_dominated", "population"],
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        linear_scaling: bool,
        evaluate_individual: callable,
        to_sympy: callable
    ):
        self.log_file = log_file

        if self.log_file is not None:
            self.log_meta = log_meta
            self.log_level = log_level

            self.X = X
            self.y = y
            self.var_y = np.var(y) + 1e-6

            self.X_test = X_test
            self.y_test = y_test
            self.var_y_test = np.var(y_test) + 1e-6 if y_test is not None else 1.0

            self.linear_scaling = linear_scaling
            self.evaluate_individual = evaluate_individual
            self.to_sympy = to_sympy

            self.meta = []
            self.indices_to_log = []

            pdir = os.path.dirname(log_file)
            if len(pdir) > 0:
                os.makedirs(pdir, exist_ok=True)
            
            self.file = open(self.log_file, "+a", encoding="utf-8")
            meta_headers = []
            for k,v in log_meta.items():
                meta_headers.append(str(k))
                self.meta.append(f'"{v}"' if isinstance(v, str) else v)
            self.file.write(",".join([
                "generation",
                "evaluations",
                "time_seconds",
                "time_seconds_raw",
                "r2_train",
                "r2_test",
                "mse_train",
                "mse_test",
                "size",
                "expression",
            ] + meta_headers) + "\n")
    
    def log(self, generation, evaluations, time_seconds, time_seconds_raw, structures, constants, fitness):
        if self.log_file is not None:
            if self.log_level == "mse_elite":
                self.indices_to_log = [fitness[:,0].argmin()]
            elif self.log_level == "non_dominated":
                self.indices_to_log = pg.non_dominated_front_2d(fitness).astype(int)
            elif self.log_level == "population" and len(self.indices_to_log) == 0:
                self.indices_to_log = list(range(structures.shape[0]))
            
            fitness_test = np.array([np.inf, np.inf], dtype=np.float32)
            for i in self.indices_to_log:
                if self.X_test is not None:
                    self.evaluate_individual(structures[i], constants[i], fitness_test, self.X_test, self.y_test, self.linear_scaling)

                self.file.write(",".join(map(str, [
                    generation,
                    evaluations,
                    f"{time_seconds:.3f}",
                    f"{time_seconds_raw:.3f}",
                    1 - fitness[i, 0] / self.var_y,
                    1 - fitness_test[0] / self.var_y_test,
                    fitness[i, 0],
                    fitness_test[0],
                    fitness[i, 1],
                    f'"{self.to_sympy(structures[i], constants[i], self.X, self.y, self.linear_scaling)}"'
                ] + self.meta)) + "\n")
    
    def __del__(self):
        if self.log_file is not None:
            self.file.close()

class Profiler:
    """A profiler using cProfile to show the most time intensive functions.
    
    Usage:
    ```
    with Profiler():
        ... do something
    ```
    """
    def __init__(self, max_rows: int = 5) -> None:
        from cProfile import Profile
        self.max_rows = max_rows
        self.p = Profile()
    
    def __enter__(self):
        self.p.__enter__()
    
    def __exit__(self, *args, **kwargs):
        self.p.__exit__(*args, **kwargs)
        import pstats

        pstats.Stats(self.p) \
            .strip_dirs() \
            .sort_stats("time") \
            .print_stats(self.max_rows)

def debug_print_jit_info(fn):
    """Prints information about an already compiled function"""
    signature = fn.signatures[0]
    overload = fn.overloads[signature]
    width = 20
    print("Signature:", signature)
    for name, t in overload.metadata["pipeline_times"]["nopython"].items():
        print(f"{name: <{40}}: {t.init:<{width}.6f}{t.run:<{width}.6f}{t.finalize:<{width}.6f}")

### variation.py ###
import numpy as np
import numba as nb
from numba import types as nty

def get_variation_fn(
    population_size: int,
    max_expression_size: int,
    num_constants: int,
    library_size: int,
    p_crossover: float,
    scaling_factor: float,
    linear_scaling: bool,
    evaluate_individual: callable,
    evaluate_population: callable
):
    @nb.jit((
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C", readonly=True),
        nty.Array(nty.float32, 1, "C", readonly=True),
        nb.typeof(np.random.Generator(np.random.Philox()))
        ), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
    def perform_variation(structures, constants, fitness, trial_structures, trial_constants, trial_fitness, X, y, rng):
        """Performs a variation step and returns the number of fitness evaluations performed."""
        for i in range(population_size):
            r0 = r1 = r2 = i
            while r0 == i:                          r0 = rng.integers(0, population_size)
            while r1 == i or r1 == r0:              r1 = rng.integers(0, population_size)
            while r2 == i or r2 == r0 or r2 == r1:  r2 = rng.integers(0, population_size)
            j_rand: np.int32  = rng.integers(0, max_expression_size + num_constants)

            # construct trial population
            for j in range(structures.shape[1]):
                if rng.random() < p_crossover or j == j_rand:
                    trial_structures[i, j] = structures[r0, j] + scaling_factor * (structures[r1, j] - structures[r2, j])
                    # repair as per Eq 8 (https://doi.org/10.1145/1389095.1389331)
                    v_abs = np.abs(trial_structures[i, j])
                    v_floored_abs = np.floor(v_abs)
                    trial_structures[i, j] = (v_floored_abs % library_size) + (v_abs - v_floored_abs)
                else:
                    trial_structures[i, j] = structures[i, j]
            
            if j_rand > max_expression_size:
                j_rand -= max_expression_size
            
            for j in range(constants.shape[1]):
                if rng.random() < p_crossover or j == j_rand:
                    trial_constants[i, j] = constants[r0, j] + scaling_factor * (constants[r1, j] - constants[r2, j])
                else:
                    trial_constants[i, j] = constants[i, j]

        evaluate_population(trial_structures, trial_constants, trial_fitness, X, y, linear_scaling)
        return population_size
    
    return perform_variation
