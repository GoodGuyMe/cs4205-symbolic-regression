import os
import sys
import importlib
import pathlib

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
    key = (f"{max_expression_size}_{num_constants}_{max_instances}_{num_inputs}_{'_'.join(operators)}"
           .replace("/", "／").replace("*", "x"))
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
            # compute the MSE as fitness
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

    # initialize an empty sym_buffer of max size
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
                # take top operator off the stack as all its arguments have been processed
                buf_idx, op, _ = op_stack.pop()
                _arity = arity[op]
                if op < num_operators:
                    # function operator, format using arguments from the sym_buffer
                    sym_buffer[buf_idx] = fmt[op] \
                        .format(*[sym_buffer[arg_stack[ai - _arity]] for ai in range(_arity)])
                else:
                    op -= num_operators

                    if op < num_inputs:
                        # input variable, store as terminal in sym_buffer
                        sym_buffer[buf_idx] = f"x{op}"
                    else:
                        # constants, retrieve value from constants list, store as terminal in sym_buffer
                        value = constants[op - num_inputs]
                        sym_buffer[buf_idx] = str(value) if precision is None or simplify else f"{value:.{precision}g}"
                # reduce the arity of the operator as one of its arguments has been processed
                arg_stack = arg_stack[:len(arg_stack) - _arity]
                if len(op_stack) > 0:
                    op_stack[-1][2] -= 1
            else:
                # add new operator to the stack
                op = int(abs(structure[j]))
                # store the position of the operator in the structure, the operator and its arity
                op_stack.append([j, op, arity[op]])
                arg_stack.append(j)
                j += 1
        if j >= max_expression_size:
            return None
        
        if linear_scaling:
            eval_buffer = np.empty(shape=(max_instances, max_expression_size), dtype=np.float32)
            # compute the final output of the expression and store in eval_buffer 
            compute_output(structure, constants, eval_buffer, X)
            eval_buffer[:X.shape[0], 1] = 1
            # determine w and b that minimize MSE 
            w, b = np.linalg.lstsq(eval_buffer[:X.shape[0], :2], y, rcond=None)[0]
            # store the scaled version in sym_buffer
            sym_buffer[0] = f"{b} + {w} * ({sym_buffer[0]})"

        if not simplify:
            return sym_buffer[0]

        # simplify the expression using the sym package
        e = sym.simplify(sym.sympify(sym_buffer[0]), ratio=1.0)
        if precision is None:
            return str(e)

        for n in sym.preorder_traversal(e):
            if isinstance(n, sym.Float):
                e = e.subs(n, sym.Float(n, precision + 1))
        return e

    return evaluate_individual, evaluate_population, to_sympy
