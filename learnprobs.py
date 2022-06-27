import sys

import numpy as np
import stormpy
import stormpy.examples
import stormpy.examples.files

import observations


def frequentist(sample: np.ndarray, model: stormpy.SparseDtmc, smoothing: float = 0) -> np.ndarray:
    """
    Learn the probabilities of a model using the frequentist way.

    When out of N samples we observe P times successor state si, we estimate:
    P(s0, a, si) ~= P/N

    :param smoothing: Laplace smoothing parameter to avoid skipping potential transitions that would just not be in
    the sample.
    :param sample: List of random observations on the model
    :param model: DTMC model which we want to learn the transition probabilities.
    """
    N_states = len(model.states)

    P = np.zeros([N_states, N_states])
    N = np.zeros([N_states])

    for (i, j) in sample:
        P[i, j] += 1
        N[i] += 1

    for iv in [(i // N_states, i % N_states) for i in range(N_states * N_states)]:
        nb_trans = len(model.states[iv[0]].actions[0].transitions)
        P[iv] = (P[iv] + smoothing) / (N[iv[0]] + nb_trans * smoothing)

    return P


if __name__ == "__main__":
    program = stormpy.parse_prism_program(stormpy.examples.files.prism_dtmc_die)
    model = stormpy.build_model(program)

    obs = observations.parse_observations(observations.DEFAULT_PATH)

    if len(sys.argv) > 1:
        method = sys.argv[1]
        if method == "frequentist":
            a = frequentist(obs, model)
