import os
import time

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

results = [[] for _ in range(len(models))]
prob_s0 = [[] for _ in range(len(models))]
sse = [[] for _ in range(len(models))]
times = []
for i in range(len(models)):
    model = models[i]
    for ob in obs[i]:
        time_frequentist = time.time()
        matrix = learnprobs.frequentist(ob, model, smoothing=1)
        time_frequentist = time.time() - time_frequentist
        m1 = learnprobs.model_from_sparse_matrix(matrix, model.labeling, model.reward_models)

        time_bayesian = time.time()
        matrix = learnprobs.bayesian_dirichlet(ob, model)
        time_bayesian = time.time() - time_bayesian
        m2 = learnprobs.model_from_sparse_matrix(matrix, model.labeling, model.reward_models)

        times.append([time_frequentist * 1000, time_bayesian * 1000])

        for prop in props[i]:
            result_base = stormpy.model_checking(model, prop)
            result_frequentist = stormpy.model_checking(m1, prop)
            result_bayesian = stormpy.model_checking(m2, prop)

            results[i].append([result_base, result_frequentist, result_bayesian])
            prob_s0[i].append([result_base.at(model.initial_states[0]),
                               result_frequentist.at(model.initial_states[0]),
                               result_bayesian.at(model.initial_states[0])])

            sse_frequentist, sse_bayesian = 0, 0
            for s in model.states:
                p_base = results[i][-1][0].at(s)
                p_frequentist = results[i][-1][1].at(s)
                p_bayesian = results[i][-1][2].at(s)

                sse_frequentist += (p_base - p_frequentist) ** 2
                sse_bayesian += (p_base - p_bayesian) ** 2

            sse[i].append([sse_frequentist, sse_bayesian])

            print("Base prob at s0: ", result_base.at(model.initial_states[0]))
            print("Frequentist prob at s0: ", result_frequentist.at(model.initial_states[0]))
            print("Bayesian prob at s0: ", result_bayesian.at(model.initial_states[0]))

# TODO: Use the reward tuples in a significant way by actually formatting data
plt.plot(prob_s0)

# TODO: generate plots and save them in the plots folder
plt.savefig(os.path.join("plots/", "plot.png"))
