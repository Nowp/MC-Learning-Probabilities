import math
import os
import sys

import numpy
import numpy as np
import stormpy
import stormpy.examples
import stormpy.examples.files

import observations


def model_from_sparse_matrix(trans_matrix: stormpy.SparseMatrix,
                             labels: stormpy.StateLabeling = None,
                             rewards: dict[stormpy.SparseRewardModel] = None) -> stormpy.SparseDtmc:
    """
    Creates a DTMC model from a given transition matrix.


    :param rewards: Dictionary of reward models
    :param trans_matrix: Sparse matrix of transition probabilities
    :param labels: StateLabeling object holding labels for each state
    """
    if rewards is None:
        rewards = {}

    n_states = trans_matrix.nr_rows

    if labels is None:
        labeling = stormpy.storage.StateLabeling(n_states)
    else:
        labeling = labels

    components = stormpy.SparseModelComponents(transition_matrix=trans_matrix)
    components.state_labeling = labeling
    components.reward_models = rewards
    dtmc = stormpy.storage.SparseMdp(components)
    return dtmc


def frequentist(sample: np.ndarray, model: stormpy.SparseDtmc, smoothing: float = 0) -> stormpy.SparseMatrix:
    """
    Learn the probabilities of a model using the frequentist way.

    When out of N samples we observe P times successor state si, we estimate:
    P(s0, a, si) ~= P/N

    :param smoothing: Laplace smoothing parameter to avoid skipping potential transitions that would just not be in
    the sample.
    :param sample: List of random observations on the model
    :param model: DTMC model which we want to learn the transition probabilities.
    """
    n_states = model.nr_states
    n_choices = [len(s.actions) for s in model.states]
    n = np.zeros([max(n_choices), n_states])
    nb_trans = np.zeros([max(n_choices), n_states], numpy.int8)

    row = np.zeros([max(n_choices), n_states + 1], numpy.int8)
    col = [[] for _ in range(max(n_choices))]
    for s in model.states:
        for act in s.actions:
            nb_trans[act.id][s.id] = len(act.transitions)
            for t in act.transitions:
                row[act.id, s.id + 1] = row[act.id, s.id] + nb_trans[act.id][s.id]
                col[act.id].append(t.column)

    values = np.zeros([max(n_choices), np.sum(nb_trans)])

    for (start, dest, choice) in sample:
        n[choice, start] += 1
        i = row[choice, start]
        for j in range(i, row[choice, start + 1]):
            if col[choice][j] == dest:
                values[choice, j] += 1

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
                values[local_choice, i] = (values[local_choice, i] + smoothing) / (n[local_choice, s] + nb_trans[local_choice, s] * smoothing)
                c = col[local_choice][i]
                v = values[local_choice, i]
                builder.add_next_value(choice, c, v)
        next_group = choice + 1

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
    model_path = sys.argv[1]
    properties_path = sys.argv[2]

    program = stormpy.parse_prism_program(model_path)
    model = stormpy.build_model(program)

    obs = observations.parse_observations(observations.DEFAULT_PATH)

    with open(properties_path, 'r') as fr:
        properties_raw = fr.readlines()

    properties = stormpy.parse_properties(';'.join(properties_raw))
    m = None

    method = sys.argv[3]
    if method == "frequentist":
        matrix = frequentist(obs, model)
    elif method == "bayesian":
        matrix = bayesian_dirichlet(obs, model)
    else:
        raise NotImplementedError("This method is not implemented")

    m = model_from_sparse_matrix(matrix, model.labeling, model.reward_models)

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

