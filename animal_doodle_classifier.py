# Google Quick Draw dataset
# Dogs doodles
# https://github.com/googlecreativelab/quickdraw-dataset

import numpy as np
from ../neural_nets/multilayer_perceptron import MutlilayerPerceptron

dog_data = np.load("doodle_dataset/dog_train_1200.npy")
fish_data = np.load("doodle_dataset/fish_train_1200.npy")
octopus_data = np.load("doodle_dataset/octopus_train_1200.npy")
penguin_data = np.load("doodle_dataset/penguin_train_1200.npy")
