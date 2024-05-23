import numpy as np

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