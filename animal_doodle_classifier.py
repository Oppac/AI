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

dog_data = np.load("doodle_dataset/full/full_dog.npy")
fish_data = np.load("doodle_dataset/full/full_fish.npy")
octopus_data = np.load("doodle_dataset/full/full_octopus.npy")
penguin_data = np.load("doodle_dataset/full/full_penguin.npy")

layers = [784, 4, 5, 2]
learning_rate = 0.5
neural_net = MutlilayerPerceptron(layers, learning_rate)

animals = ['d', 'f', 'o', 'p']
indexes = [0, 0, 0, 0]

for i in range(len(dog_data)):
    animal_rand = random.choice(animals)
    if animal_rand == 'd':
        indexes[0] += 1
        input_rand = vectorize(dog_data[indexes[0]])
        output_rand = [ [1], [0], [0], [0] ]
    elif animal_rand == 'f':
        indexes[1] += 1
        input_rand = vectorize(fish_data[indexes[1]])
        output_rand = [ [0], [1], [0], [0] ]
    elif animal_rand == 'o':
        indexes[2] += 1
        input_rand = vectorize(octopus_data[indexes[2]])
        output_rand = [ [0], [0], [1], [0] ]
    elif animal_rand == 'p':
        indexes[3] += 1
        input_rand = vectorize(penguin_data[indexes[3]])
        output_rand = [ [0], [0], [0], [1] ]
    neural_net.train(input_rand, output_rand)

inputs = Matrix(784, 1)
inputs.give_values(vectorize(dog_data[0]))
print(neural_net.feedforward(inputs)[-1].values)
