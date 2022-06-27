import doctest

doctest.ELLIPSIS_MARKER = '-etc-'
import stormpy
import stormpy.examples
import stormpy.examples.files

import numpy as np

N = 1000
DEFAULT_PATH = "observations.dat"

program = stormpy.parse_prism_program(stormpy.examples.files.prism_dtmc_die)
model = stormpy.build_model(program)


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
                transitions.append((state.id, transition.column))

    with open(path, "w") as fw:
        obs = [transitions[i] for i in np.random.randint(0, len(transitions) - 1, size)]
        for (s, d) in obs:
            fw.write(f"{s} {d}\n")


def parse_observations(path: str) -> np.ndarray:
    """
    Reads a list of observations from a given file.
    :param path: Destination of the file with the observations.
    :return: 1-D Array of tuple with each element like (Start State, Destination State)
    """
    with open(path, "r") as fr:
        lines = fr.readlines()
        lines = [newline.split(" ") for newline in lines]
        lines = [[int(val) for val in newline] for newline in lines]

    return np.array(lines)