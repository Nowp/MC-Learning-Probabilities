import os
import numpy.random
import stormpy
import learnprobs
import observations
import matplotlib.pyplot as plt
import random

os.makedirs("models", exist_ok=True)
os.makedirs("observations", exist_ok=True)
os.makedirs("properties", exist_ok=True)
os.makedirs("plots", exist_ok=True)

seed_number = 42
random.seed(seed_number)
numpy.random.seed(seed_number)

models = []
props = []
nb_vals = [100, 1000, 5000, 10000]

for model_path in os.listdir("models"):
    program = stormpy.parse_prism_program(os.path.join("models", model_path))
    models.append(stormpy.build_model(program))

obs = [[] for _ in range(len(models))]

for N in nb_vals:
    for model_index in range(len(models)):
        obs_path = os.path.join("observations", f"observation-{model_index}-{N}")
        observations.gen_observations(models[model_index], N, obs_path)
        obs[model_index].append(observations.parse_observations(obs_path))

for prop_path in os.listdir("properties"):
    with open(os.path.join("properties", prop_path), 'r') as fr:
        properties_raw = fr.readlines()
        props.append(stormpy.parse_properties(';'.join(properties_raw)))

result_tuples = []
reward_tuples = []

for i in range(len(models)):
    model = models[i]
    for prop in props[i]:
        for ob in obs[i]:
            matrix = learnprobs.frequentist(ob, model)
            m1 = learnprobs.model_from_sparse_matrix(matrix, model.labeling, model.reward_models)

            matrix = learnprobs.bayesian_dirichlet(ob, model)
            m2 = learnprobs.model_from_sparse_matrix(matrix, model.labeling, model.reward_models)

            result_base = stormpy.model_checking(model, prop)
            result_frequentist = stormpy.model_checking(m1, prop)
            result_bayesian = stormpy.model_checking(m2, prop)

            # TODO: figure out what to do with these
            # Maybe MSE of state-action reward could be used (estimation vs base)?
            # Or/And a simple reward aggregate vs base?
            # We could also do a time-based analysis by adding timers.

            result_tuples.append((result_base, result_frequentist, result_bayesian))
            reward_tuples.append((result_base.at(model.initial_states[0]),
                                  result_frequentist.at(model.initial_states[0]),
                                  result_bayesian.at(model.initial_states[0])))

            print("Base reward: ", result_base.at(model.initial_states[0]))
            print("Frequentist reward: ", result_frequentist.at(model.initial_states[0]))
            print("Bayesian reward: ", result_bayesian.at(model.initial_states[0]))

# TODO: Use the reward tuples in a significant way by actually formatting data
plt.plot(reward_tuples)

# TODO: generate plots and save them in the plots folder
plt.savefig(os.path.join("plots/", "plot.png"))