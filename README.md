# Model-Checking-Learning-Probabilities
This project is our implementation in Python with Stormpy for learning the probabilities of a DTMC by using statistical methods.
For generate_plots.py, we decided to choose the Knuth-Yao-Die MDP and the Maze MDP example found in the Storm examples with several formulae that can be found in the "properties" folder.

The observation sample used is a list composed of N (100, 1000 & 10000) randomly chosen transitions from the original model while taking into account the transition's probability, as if we observed N times any of these transitions with a uniform distribution.

To compare the results of these different models, we look at what their evaluations are for all of these properties and how close they are from the original model.
We also compare the learning time of the Frequentist and Bayesian methods on the different sample sizes.

# Running the project

Running generate_plots.py will automatically generate all the plots for the models in the models folder and the properties in the properties folder. If you want to run our code on other models, you can add a model and a property file to both folders with the same name (eg: dice.nm for the model and dice.prop for the properties). Each property must be on a new line (look at the pre-existing files for more information) and will be ran with each of the preset sample sizes. If you want to change the sample sizes you can modify the list at the beginning of generate_plots.py.
