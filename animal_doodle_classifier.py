# Google Quick Draw dataset
# Dogs doodles
# https://github.com/googlecreativelab/quickdraw-dataset

import random
import numpy as np
from multilayer_perceptron import MutlilayerPerceptron
from matrix import Matrix

def vectorize(data):
    vector_data = []
    for i in data:
        if i > 0:
            vector_data.append([1])
        else:
            vector_data.append([0])
    return vector_data

fish_data = np.load("doodle_dataset/fish_train_1200.npy")
fish_testing = np.load("doodle_dataset/fish_testing_200.npy")

octopus_data = np.load("doodle_dataset/octopus_train_1200.npy")
octopus_testing = np.load("doodle_dataset/octopus_testing_200.npy")

layers = [784, 7, 1, 2]
learning_rate = 0.1
neural_net = MutlilayerPerceptron(layers, learning_rate)

animals = ['f', 'o']
indexes = [0, 0]

for i in range(len(fish_data)):
    animal_rand = random.choice(animals)
    if animal_rand == 'f':
        indexes[0] += 1
        input_rand = vectorize(fish_data[indexes[0]])
        output_rand = [ [1], [0] ]
    elif animal_rand == 'o':
        indexes[1] += 1
        input_rand = vectorize(octopus_data[indexes[1]])
        output_rand = [ [0], [1] ]
    neural_net.train(input_rand, output_rand)

inputs = Matrix(784, 1)
size = 100
fish_score = 0
octopus_score = 0

for i in range(size):
    inputs.give_values(vectorize(fish_testing[i]))
    result = neural_net.feedforward(inputs)[-1].values
    if result[0][0] > result[1][0]:
        fish_score += 1

for i in range(size):
    inputs.give_values(vectorize(octopus_testing[i]))
    result = neural_net.feedforward(inputs)[-1].values
    if result[1][0] > result[0][0]:
        octopus_score += 1
print()
print("Fish accuracy: " + str(fish_score) + "%")
print("Octopus accuracy: " + str(octopus_score) + "%")
