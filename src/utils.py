from typing import Literal
import re
import os
import string
import enum

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