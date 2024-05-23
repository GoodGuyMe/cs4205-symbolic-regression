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
