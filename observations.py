import doctest
doctest.ELLIPSIS_MARKER = '-etc-'
import stormpy
import stormpy.examples
import stormpy.examples.files

import numpy as np

N = 1000
DEFAULT_PATH = "observations.dat"


def gen_observations(model: stormpy.SparseDtmc, size: int = N, path=DEFAULT_PATH) -> None:
    """
    Generates a file with N observations of transitions of the given DTMC.

    This is simply creating a list containing N randomly chosen elements from the set of all possible transitions.
    :param path: Destination file of the generated observations.
    :param model: DTMC where the observations will be made.
    :param size: Number of observations to make.
    """
    transitions = []
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                transitions.append((state.id, transition.value(), transition.column))

    with open(path, "w") as fw:
        obs = [transitions[i] for i in np.random.randint(0, len(transitions)-1, size)]
        for (s, v, d) in obs:
            fw.write(f"{s} {v} {d}\n")
