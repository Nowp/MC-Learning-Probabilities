import doctest
doctest.ELLIPSIS_MARKER = '-etc-'
import stormpy
import stormpy.examples
import stormpy.examples.files

import numpy as np

program = stormpy.parse_prism_program(stormpy.examples.files.prism_dtmc_die)
model = stormpy.build_model(program)

N = 1000

transitions = []

for state in model.states:
    for action in state.actions:
        for transition in action.transitions:
            transitions.append((state.id, transition.value(), transition.column))

with open("sequences.dat", "a") as fw:
    obs = [transitions[i] for i in np.random.randint(0, len(transitions)-1, N)]
    for (s, v, d) in obs:
        fw.write(f"{s} {v} {d}\n")
