import numpy as np
import numba as nb
from numba import types as nty
import math


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
        epsilon: float,
        structure_search: str,
        constants_search: str,
        search_perc: int,
        consts_diff: bool,
        search_threshold: int,
        num_operators: int
):
    # @nb.jit((
    #         nty.Array(nty.float32, 2, "C"),
    #         nty.Array(nty.float32, 2, "C"),
    #         nty.Array(nty.float32, 2, "C"),
    #         nty.Array(nty.float32, 2, "C"),
    #         nty.Array(nty.float32, 2, "C"),
    #         nty.Array(nty.float32, 2, "C"),
    #         nty.Array(nty.float32, 2, "C", readonly=True),
    #         nty.Array(nty.float32, 1, "C", readonly=True),
    #         nb.typeof(np.random.Generator(np.random.Philox())),
    #         nty.float32,
    #         nty.float32
    # ), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=False,
    #     parallel=False)
    def perform_variation(structures, constants, fitness, trial_structures, trial_constants, trial_fitness, X, y, rng,
                          prev_best_fit, learning_rate):
        """Performs a variation step and returns the number of fitness evaluations performed."""
        iteration = 0

        def get_constants_used(structure):
            consts_used = []
            for struct in structure:
                op = int(abs(struct))
                if op >= (num_operators + X.shape[1]):
                    op -= num_operators + X.shape[1]
                    consts_used.append(op)
            # print(consts_used)
            return consts_used

        def update_parameters(param, grad, temp_param, lr):
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param[i, j] = temp_param[i, j] + (grad[i, j] * lr)

        def finite_differencing(X, finite_difference_method, fitness, to_edit, to_leave, y, edit_structs, best_idxs):
            gradients = np.zeros_like(to_edit)
            for i in best_idxs:
                to_edit_used = []
                if not edit_structs:
                    to_edit_used = get_constants_used(to_leave[i])
                else:
                    to_edit_used = to_edit.shape[1]
                for j in to_edit_used:
                    # Perturb the structure/constants positively and negatively
                    original_value = to_edit[i, j]

                    # f'(x) = 1/h * (f(x + h) - f(x)) : forward difference method
                    # f'(x) = 1/h * (f(x) - f(x - h)) : backward difference method
                    # f'(x) = 1/(2h) * (f(x + h) - f(x - h)) : central difference method

                    # compute first evaluation
                    if finite_difference_method == 'forward' or finite_difference_method == 'central':
                        to_edit[i, j] = original_value + epsilon

                    if edit_structs:
                        evaluate_individual(to_edit[i], to_leave[i], fitness[i], X, y, linear_scaling)
                    else:

                        evaluate_individual(to_leave[i], to_edit[i], fitness[i], X, y, linear_scaling)

                    # Get MSE
                    fitness_first = fitness[i][0]

                    # reset to original value
                    to_edit[i, j] = original_value

                    # compute second evaluation
                    if finite_difference_method == 'backward' or finite_difference_method == 'central':
                        to_edit[i, j] = original_value - epsilon

                    if edit_structs:
                        evaluate_individual(to_edit[i], to_leave[i], fitness[i], X, y, linear_scaling)
                    else:
                        evaluate_individual(to_leave[i], to_edit[i], fitness[i], X, y, linear_scaling)

                    # Get MSE
                    fitness_second = fitness[i][0]

                    # reset to original value
                    to_edit[i, j] = original_value

                    # Calculate the gradient using finite differences
                    multiplier = 2 if finite_difference_method == 'central' else 1
                    if not math.isinf(fitness_first) and not math.isinf(fitness_second):
                        gradients[i, j] = ((fitness_first - fitness_second) / (multiplier * epsilon))
            return gradients

        temp_trial_structs = np.zeros_like(trial_structures)
        temp_trial_fit = np.zeros_like(trial_fitness)
        temp_trial_consts = np.zeros_like(trial_constants)

        ############################################ MUTATION ################################################

        for i in range(population_size):
            r0 = r1 = r2 = i
            while r0 == i:                          r0 = rng.integers(0, population_size)
            while r1 == i or r1 == r0:              r1 = rng.integers(0, population_size)
            while r2 == i or r2 == r0 or r2 == r1:  r2 = rng.integers(0, population_size)
            # randomly select one index for which mutation takes place for sure
            j_rand: np.int32 = rng.integers(0, max_expression_size + num_constants)

            # construct trial population
            for j in range(structures.shape[1]):
                # perform crossover on selected index j_rand with proba 1 or with proba p_crossover on other indices
                if rng.random() < p_crossover or j == j_rand:
                    temp_trial_structs[i, j] = structures[r0, j] + scaling_factor * (
                            structures[r1, j] - structures[r2, j])
                    # repair as per Eq 8 (https://doi.org/10.1145/1389095.1389331) to arive at an integer that corresponds to an operator or variable
                    v_abs = np.abs(temp_trial_structs[i, j])
                    v_floored_abs = np.floor(v_abs)
                    temp_trial_structs[i, j] = (v_floored_abs % library_size) + (v_abs - v_floored_abs)
                else:
                    temp_trial_structs[i, j] = structures[i, j]

            if j_rand > max_expression_size:
                j_rand -= max_expression_size

            for j in range(constants.shape[1]):
                # perform crossover on selected index j_rand with proba 1 or with proba p_crossover on other indices
                if consts_diff and rng.random() < p_crossover or j == j_rand:
                    # as these are constants, no repair is required
                    temp_trial_consts[i, j] = constants[r0, j] + scaling_factor * (
                                constants[r1, j] - constants[r2, j])
                else:
                    temp_trial_consts[i, j] = constants[i, j]

        ################################ IDENTIFY LOCAL SEARCH CANDIDATES ####################################

        apply_local_search = False
        num_solutions = temp_trial_fit.shape[0]
        best_indices = np.arange(num_solutions)
        if constants_search != 'none':
            evaluate_population(temp_trial_structs, temp_trial_consts, temp_trial_fit, X, y, linear_scaling)
            # Sort the population based on fitness proceed with local search for the best search_perc solutions
            best_indices = np.argsort(temp_trial_fit[:, 0])[:min(num_solutions, int(np.ceil(num_solutions * search_perc / 100)))]
            # Only apply local search when the search_threshold is exceeded

            new_best_fit = temp_trial_fit[temp_trial_fit[:, 0].argmin(), 0]
            apply_local_search = (new_best_fit - prev_best_fit) / prev_best_fit * 100 < -search_threshold

        ########################################## LOCAL SEARCH ##############################################

        structs_grad = np.zeros_like(trial_structures)
        if structure_search != 'none':
            structs_grad = finite_differencing(X, structure_search, temp_trial_fit, temp_trial_structs,
                                               temp_trial_consts, y, True, best_indices)

        consts_grad = np.zeros_like(trial_constants)
        if constants_search != 'none' and apply_local_search:
            consts_grad = finite_differencing(X, constants_search, temp_trial_fit, temp_trial_consts,
                                              temp_trial_structs, y, False, best_indices)

        ############################################ UPDATE ################################################

        # update the parameters
        update_parameters(trial_structures, structs_grad, temp_trial_structs, learning_rate)
        update_parameters(trial_constants, consts_grad, temp_trial_consts, learning_rate)

        # evaluate the population
        evaluate_population(trial_structures, trial_constants, trial_fitness, X, y, linear_scaling)
        return population_size

    return perform_variation
