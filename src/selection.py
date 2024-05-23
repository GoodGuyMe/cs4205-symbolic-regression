import numpy as np
import numba as nb
from numba import types as nty

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
