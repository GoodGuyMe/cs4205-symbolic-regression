import numpy as np
import numba as nb
from numba import types as nty
import math

iteration = 0

def get_variation_fn(
    population_size: int,
    max_expression_size: int,
    num_constants: int,
    library_size: int,
    p_crossover: float,
    scaling_factor: float,
    linear_scaling: bool,
    evaluate_individual: callable,
    evaluate_population: callable,
    initial_learning_rate: float = 0.01,
    learning_rate_decay: float = 0.99,

    epsilon: float = 0.00001
):

    def update_learning_rate():
        global iteration
        iteration += 1
        print(iteration, flush=True)
        return initial_learning_rate * (learning_rate_decay ** iteration)

    # @nb.jit((
    #     nty.Array(nty.float32, 2, "C"),
    #     nty.Array(nty.float32, 2, "C"),
    #     nty.Array(nty.float32, 2, "C"),
    #     nty.Array(nty.float32, 2, "C"),
    #     nty.Array(nty.float32, 2, "C"),
    #     nty.Array(nty.float32, 2, "C"),
    #     nty.Array(nty.float32, 2, "C", readonly=True),
    #     nty.Array(nty.float32, 1, "C", readonly=True),
    #     nb.typeof(np.random.Generator(np.random.Philox()))
    #     ), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
    def perform_variation(structures, constants, fitness, trial_structures, trial_constants, trial_fitness, X, y, rng):
        """Performs a variation step and returns the number of fitness evaluations performed."""

        # Implement gradient-based local search to optimize the structures of a genetic programming population using gradient with finite differences

        temp_trial_structures = np.zeros_like(trial_structures)
        temp_trial_fitness = np.zeros_like(trial_structures)

        learning_rate = update_learning_rate()


        for i in range(population_size):
            r0 = r1 = r2 = i
            while r0 == i:                          r0 = rng.integers(0, population_size)
            while r1 == i or r1 == r0:              r1 = rng.integers(0, population_size)
            while r2 == i or r2 == r0 or r2 == r1:  r2 = rng.integers(0, population_size)
            # randomly select one index for which mutation takes place for sure
            j_rand: np.int32  = rng.integers(0, max_expression_size + num_constants)

            # construct trial population
            for j in range(structures.shape[1]):
                # perform crossover on selected index j_rand with proba 1 or with proba p_crossover on other indices
                if rng.random() < p_crossover or j == j_rand:
                    temp_trial_structures[i, j] = structures[r0, j] + scaling_factor * (structures[r1, j] - structures[r2, j])
                    # repair as per Eq 8 (https://doi.org/10.1145/1389095.1389331) to arive at an integer that corresponds to an operator or variable
                    v_abs = np.abs(temp_trial_structures[i, j])
                    v_floored_abs = np.floor(v_abs)
                    temp_trial_structures[i, j] = (v_floored_abs % library_size) + (v_abs - v_floored_abs)
                else:
                    temp_trial_structures[i, j] = structures[i, j]

            if j_rand > max_expression_size:
                j_rand -= max_expression_size

            for j in range(constants.shape[1]):
                # perform crossover on selected index j_rand with proba 1 or with proba p_crossover on other indices
                if rng.random() < p_crossover or j == j_rand:
                    # as these are constants, no repair is required
                    trial_constants[i, j] = constants[r0, j] + scaling_factor * (constants[r1, j] - constants[r2, j])
                else:
                    trial_constants[i, j] = constants[i, j]

        # evaluate_population(temp_trial_structures, trial_constants, temp_trial_fitness, X, y, linear_scaling)
        finite_difference_method = 'central'  # 'forward' or 'central' or 'backward'

        gradients = np.zeros_like(trial_structures)
        for i in range(trial_structures.shape[0]):
            for j in range(trial_structures.shape[1]):
                # Perturb the structure positively and negatively
                if finite_difference_method == 'forward':
                    # f'(x) = 1/h * (f(x + h) - f(x)) : forward difference method
                    original_value = temp_trial_structures[i, j]
                    temp_trial_structures[i, j] = original_value + epsilon
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_plus = temp_trial_fitness[i][0]
                    temp_trial_structures[i, j] = original_value
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_minus = temp_trial_fitness[i][0]
                    # Calculate the gradient using finite differences
                    if not math.isinf(fitness_plus) and not math.isinf(fitness_minus):
                        gradients[i, j] = ((fitness_plus - fitness_minus) / (epsilon))
                    # else:
                    #     print("inf")
                    # Reset the structure to its original value
                    temp_trial_structures[i, j] = original_value
                elif finite_difference_method == 'backward':
                    # f'(x) = 1/h * (f(x) - f(x - h)) : backward difference method
                    original_value = temp_trial_structures[i, j]
                    temp_trial_structures[i, j] = original_value
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_plus = temp_trial_fitness[i][0]
                    temp_trial_structures[i, j] = original_value - epsilon
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_minus = temp_trial_fitness[i][0]
                    # Calculate the gradient using finite differences
                    if not math.isinf(fitness_plus) and not math.isinf(fitness_minus):
                        gradients[i, j] = ((fitness_plus - fitness_minus) / (epsilon))
                    # else:
                    #     print("inf")
                    # Reset the structure to its original value
                    temp_trial_structures[i, j] = original_value

                elif finite_difference_method == 'central':
                    # f'(x) = 1/(2h) * (f(x + h) - f(x - h)) : central difference method
                    original_value = temp_trial_structures[i, j]
                    temp_trial_structures[i, j] = original_value + epsilon
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_plus = temp_trial_fitness[i][0]
                    temp_trial_structures[i, j] = original_value - epsilon
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_minus = temp_trial_fitness[i][0]
                    # Calculate the gradient using finite differences
                    if not math.isinf(fitness_plus) and not math.isinf(fitness_minus):
                        gradients[i, j] = ((fitness_plus - fitness_minus) / (2 * epsilon))
                    # else:
                    #     print("inf")
                    # Reset the structure to its original value
                    temp_trial_structures[i, j] = original_value
                else: # default to central difference method
                    # f'(x) = 1/(2h) * (f(x + h) - f(x - h)) : central difference method
                    original_value = temp_trial_structures[i, j]
                    temp_trial_structures[i, j] = original_value + epsilon
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_plus = temp_trial_fitness[i][0]
                    temp_trial_structures[i, j] = original_value - epsilon
                    evaluate_individual(temp_trial_structures[i], trial_constants[i], temp_trial_fitness[i], X, y, linear_scaling)
                    fitness_minus = temp_trial_fitness[i][0]
                    # Calculate the gradient using finite differences
                    if not math.isinf(fitness_plus) and not math.isinf(fitness_minus):
                        gradients[i, j] = ((fitness_plus - fitness_minus) / (2 * epsilon))
                    # else:
                    #     print("inf")
                    # Reset the structure to its original value
                    temp_trial_structures[i, j] = original_value
            

        # Use the gradients to perform a local search on the structures
        # if np.sum(gradients) != 0:
        #     print("working")

        for i in range(trial_structures.shape[0]):
            for j in range(trial_structures.shape[1]):
                trial_structures[i, j] = temp_trial_structures[i, j] + (gradients[i, j] * learning_rate)  # learning_rate is a hyperparameter to be defined

        evaluate_population(trial_structures, trial_constants, trial_fitness, X, y, linear_scaling)
        return population_size

    return perform_variation



