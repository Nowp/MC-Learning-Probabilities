import os
import sys

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
                transitions.append((state.id, transition.column, action.id, transition.value()))

    with open(path, "w") as fw:
        obs = []
        while len(obs) < size:
            (s, d, a, p) = transitions[np.random.randint(len(transitions))]
            if np.random.random() < p:
                obs.append((s, d, a))

        for (s, d, c) in obs:
            fw.write(f"{s} {d} {c}\n")


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


if __name__ == "__main__":
    model_path = sys.argv[1]
    nb_values = N
    if len(sys.argv) > 2:
        nb_values = int(sys.argv[2])
    program = stormpy.parse_prism_program(model_path)
    model = stormpy.build_model(program)

    gen_observations(model, nb_values)