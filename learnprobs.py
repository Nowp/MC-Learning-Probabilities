import math
import os
import sys
import time

import numpy
import numpy as np
import stormpy
import stormpy.examples
import stormpy.examples.files

import observations


def model_from_sparse_matrix(trans_matrix: stormpy.SparseMatrix, labels: stormpy.StateLabeling = None) -> stormpy.SparseDtmc:
    """
    Creates a DTMC model from a given transition matrix.

    :param values: Value array (Number of Non-Zero elements)
    :param col: Col array (Number of Non-Zero elements)
    :param row: Row array (Number of states elements + 1)
    :param labels: Dictionary of String -> BitVector used to assign label to each state
    """
    N_states = trans_matrix.nr_rows

    if labels is None:
        labeling = stormpy.storage.StateLabeling(N_states)
    else:
        labeling = labels

    components = stormpy.SparseModelComponents(transition_matrix=trans_matrix)
    components.state_labeling = labeling
    dtmc = stormpy.storage.SparseMdp(components)
    return dtmc


def frequentist(sample: np.ndarray, model: stormpy.SparseDtmc, coupling: (np.ndarray, [[int, int, int]]) = None, smoothing: float = 0) -> stormpy.SparseMatrix:
    """
    Learn the probabilities of a model using the frequentist way.

    When out of N samples we observe P times successor state si, we estimate:
    P(s0, a, si) ~= P/N

    :param smoothing: Laplace smoothing parameter to avoid skipping potential transitions that would just not be in
    the sample.
    :param sample: List of random observations on the model
    :param model: DTMC model which we want to learn the transition probabilities.
    """
    start_time = time.perf_counter()
    n_states = model.nr_states
    n_choices = [len(s.actions) for s in model.states]
    max_choices = max(n_choices)

    if coupling is None:
        coupling_matrix = np.zeros([max_choices, n_states, n_states])
        coupling_delegates = []
    else:
        coupling_matrix, coupling_delegates = coupling

    n = np.zeros([max_choices, n_states])
    nb_trans = np.zeros([max_choices, n_states], numpy.int8)

    row = np.zeros([max_choices, n_states + 1], numpy.int8)
    col = [[] for _ in range(max_choices)]
    for s in model.states:
        for act in s.actions:
            nb_trans[act.id][s.id] = len(act.transitions)
            for t in act.transitions:
                row[act.id, s.id + 1] = row[act.id, s.id] + nb_trans[act.id][s.id]
                col[act.id].append(t.column)

    values = np.zeros([max_choices, np.sum(nb_trans)])

    delegate_pos = [-1 for _ in range(len(coupling_delegates))]

    for (start, dest, choice) in sample:
        coupling_group = coupling_matrix[choice, start, dest] - 1
        if coupling_group == -1 or (start, dest, choice) == coupling_delegates[coupling_group][0]:
            n[choice, start] += 1
            i = row[choice, start]
            for j in range(i, row[choice, start + 1]):
                if col[choice][j] == dest:
                    values[choice, j] += 1
                    if coupling_group >= 0:
                        delegate_pos[coupling_group] = j

    # estimate p with the mode
    choice = 0
    next_group = 0
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
    for s in range(n_states):
        builder.new_row_group(next_group)
        for choice in range(next_group, next_group+n_choices[s]):
            local_choice = choice - next_group
            start, end = row[local_choice, s], row[local_choice, s + 1]
            for i in range(start, end):
                c = col[local_choice][i]
                coupling_group = coupling_matrix[local_choice, s, c] - 1
                if coupling_group == -1 or (s, c, choice) == coupling_delegates[coupling_group][0]:
                    values[local_choice, i] = (values[local_choice, i] + smoothing) / (n[local_choice, s] + nb_trans[local_choice, s] * smoothing)
                    v = values[local_choice, i]
                else:
                    v = values[coupling_delegates[coupling_group][0][2], delegate_pos[coupling_group]]

                builder.add_next_value(choice, c, v)
        next_group = choice + 1

    print(f"Total time :{time.perf_counter() - start_time}")

    return builder.build()


def bayesian_dirichlet(sample: np.ndarray, model: stormpy.SparseDtmc):
    """
    Estimates the transition probabilities using a dirichlet distribution via MAP-estimation.
    :param sample: List of random observations on the model
    :param model: Model whose transitions must be estimated
    """
    n_states = model.nr_states
    m = model.nr_transitions
    n_choices = [len(s.actions) for s in model.states]
    nb_trans = np.zeros([max(n_choices), n_states])

    row = np.zeros([max(n_choices), n_states + 1], numpy.int8)

    col = [[] for _ in range(max(n_choices))]
    for s in model.states:
        for act in s.actions:
            nb_trans[act.id][s.id] = len(act.transitions)
            for t in act.transitions:
                col[act.id].append(t.column)
                row[act.id, s.id + 1] = row[act.id, s.id] + nb_trans[act.id][s.id]

    values = np.zeros([max(n_choices), m])
    a = np.ones([max(n_choices), m])

    # Count N and k's values
    for (start, dest, choice) in sample:
        i = row[choice, start]
        for j in range(i, row[choice, start + 1]):
            if col[choice][j] == dest:
                a[choice, j] += 1

    # estimate p with the mode
    choice = 0
    next_group = 0
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
    for s in range(n_states):
        builder.new_row_group(next_group)
        for choice in range(next_group, next_group+n_choices[s]):
            local_choice = choice - next_group
            start, end = row[local_choice, s], row[local_choice, s + 1]
            for i in range(start, end):
                values[local_choice, i] = (a[local_choice, i] - 1) / (sum(a[local_choice, start:end]) - nb_trans[local_choice, s]) if a[local_choice, i] > 1 else 0
                c = col[local_choice][i]
                v = values[local_choice, i]
                builder.add_next_value(choice, c, v)
        next_group = choice + 1

    return builder.build()


if __name__ == "__main__":
    program = stormpy.parse_prism_program(os.path.join(stormpy.examples.files.testfile_dir, "mdp", "die_selection.nm"))
    model = stormpy.build_model(program)

    obs = observations.parse_observations(observations.DEFAULT_PATH)

    properties_raw = [
        'Pmax=? [F "one"]',
        'Pmax=? [F "two"]',
        'Pmax=? [F "three"]',
        'Pmax=? [F "one" | "two" | "three"]',
        'Pmax=? [G F "one"]',
        'Pmax=? [G F "two"]',
        'Pmax=? [G F "three"]',
    ]

    properties = stormpy.parse_properties(';'.join(properties_raw))
    m = None

    coupling_matrix = np.zeros([3, 43, 43], dtype=numpy.int8)
    coupling_matrix[2, 0, 1] = 1
    coupling_matrix[1, 2, 5] = 1
    coupling_matrix[2, 3, 1] = 1
    coupling_matrix[0, 0, 2] = 2
    coupling_matrix[0, 1, 4] = 2
    coupling_matrix[0, 2, 6] = 2

    coupling_delegates = [
        [(0, 1, 2), (2, 5, 1), (3, 1, 2)],
        [(0, 2, 0), (1, 4, 0), (2, 6, 0)]
    ]

    if len(sys.argv) > 1:
        method = sys.argv[1]
        if method == "frequentist":
            matrix = frequentist(obs, model)
        elif method == "bayesian":
            matrix = bayesian_dirichlet(obs, model)
        else:
            raise NotImplementedError("This method is not implemented")

        m = model_from_sparse_matrix(matrix, model.labeling)

        for state in m.states:
            for action in state.actions:
                for transition in action.transitions:
                    print(f"Action {action.id}: {state.id}, {state.labels}, {transition.value()}, {transition.column}")

        for p in range(len(properties)):
            result_base = stormpy.model_checking(model, properties[p])
            result_base_vector = [x for x in result_base.get_values()]

            result_predict = stormpy.model_checking(m, properties[p])
            result_predict_vector = [x for x in result_predict.get_values()]

            print(f"{p} : \n\tBase:\t{result_base_vector}\n\tPrediction:\t\t{result_predict_vector}")

