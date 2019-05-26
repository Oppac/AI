# Google Quick Draw dataset
# Dogs doodles
# https://github.com/googlecreativelab/quickdraw-dataset

# Classify the doodle of animals using the neural net made from scratch
# Three animals are implemented: the fish, the octopus and the owl
# Expected accurency is around 80%

# TODO -> Load models instead of training each time

import random
import time
import numpy as np
from multilayer_perceptron import MutlilayerPerceptron
from matrix import Matrix

# The images of the animals, by default in the doodle_dataset folder
# Training set: 1200 images per animal
# Testing set: 200 images per animal

fish_data = np.load("doodle_dataset/fish_train_1200.npy")
fish_testing = np.load("doodle_dataset/fish_testing_200.npy")

octopus_data = np.load("doodle_dataset/octopus_train_1200.npy")
octopus_testing = np.load("doodle_dataset/octopus_testing_200.npy")

owl_data = np.load("doodle_dataset/owl_train_1200.npy")
owl_testing = np.load("doodle_dataset/owl_testing_200.npy")

# Convert the image to a black and white array
def vectorize(data):
    vector_data = []
    for i in data:
        if i > 0:
            vector_data.append([1])
        else:
            vector_data.append([0])
    return vector_data

def create_training_data(sample_size, save=False):
    # Used to randomize the data order
    animals = ['fish', 'octo', 'owl']
    #Keep track of the number of images used for each animal
    indexes = [0, 0, 0]
    training_set = []

    for i in range(sample_size):
        animal_rand = random.choice(animals)
        data_matrix = Matrix(); label_matrix = Matrix()
        if animal_rand == 'fish':
            indexes[0] += 1
            data_matrix.give_values(vectorize(fish_data[indexes[0]]))
            label_matrix.give_values([ [1], [0], [0] ])
        elif animal_rand == 'octo':
            indexes[1] += 1
            data_matrix.give_values(vectorize(octopus_data[indexes[1]]))
            label_matrix.give_values([ [0], [1], [0] ])
        elif animal_rand == 'owl':
            indexes[2] += 1
            data_matrix.give_values(vectorize(owl_data[indexes[2]]))
            label_matrix.give_values([ [0], [0], [1] ])
        new_data = [data_matrix, label_matrix]
        training_set.append(new_data)
    if save:
        np.save("doodle_dataset/training_set.npy", training_set)
    return training_set


def main():
    # Inputs: 784 -> one for each pixel in the image
    # Hidden: 15 nodes and only one layer
    # Ouputs: 3 -> one for each class of animal: fish, octopus, owl)
    layers = [784, 15, 1, 3]
    learning_rate = 0.1
    # Number of data in the training set
    sample_size = 1500
    # The size of each data batches
    batch_size = 10
    # Number of passes through the training dataset
    epochs = 1
    neural_net = MutlilayerPerceptron(layers, learning_rate, batch_size, epochs)

    start_time = time.time()

    if 1:
        training_set = create_training_data(sample_size, True)
    else:
        training_set = owl_data = np.load("doodle_dataset/training_set.npy")


    neural_net.train(training_set)
    print("\nTraining time: " + str(time.time() - start_time) + " seconds")


    # Testing the accuracy of the training on 100 images for each animal
    inputs = Matrix(784, 1)
    size = 100
    score = [0, 0, 0]

    for i in range(size):
        inputs.give_values(vectorize(fish_testing[i]))
        result = neural_net.feedforward(inputs)[-1].values
        if result[0][0] > result[1][0] and result[0][0] > result[2][0]:
            score[0] += 1

    for i in range(size):
        inputs.give_values(vectorize(octopus_testing[i]))
        result = neural_net.feedforward(inputs)[-1].values
        if result[1][0] > result[0][0] and result[1][0] > result[2][0]:
            score[1] += 1

    for i in range(size):
        inputs.give_values(vectorize(owl_testing[i]))
        result = neural_net.feedforward(inputs)[-1].values
        if result[2][0] > result[0][0] and result[2][0] > result[1][0]:
            score[2] += 1

    print()
    #print(indexes)
    print()
    print("Fish accuracy: " + str(score[0]) + "%")
    print("Octopus accuracy: " + str(score[1]) + "%")
    print("Owl accuracy: " + str(score[2]) + "%")
    total = sum(score) // 3
    print("Total accuracy: " + str(total) + "%\n")

    # Allow to test new images on the trained network
    # Images are taken from what remain of the testing set
    # "quit" or "q" to exit
    while False:
        animal = input("Fish | Octopus | Owl: ").lower()
        if animal == "q" or animal == "quit":
            break

        img_nb = int(input("Image number [0-100]: ")) + 100
        test = Matrix(784, 1)

        if animal == "fish":
            test.give_values(vectorize(fish_testing[img_nb]))
        elif animal == "octopus":
            test.give_values(vectorize(octopus_testing[img_nb]))
        elif animal == "owl":
            test.give_values(vectorize(owl_testing[img_nb]))
        result = neural_net.feedforward(test)[-1].values
        if result[0][0] > result[1][0] and result[0][0] > result[2][0]:
            ani = "fish"
        elif result[1][0] > result[0][0] and result[1][0] > result[2][0]:
            ani = "octopus"
        elif result[2][0] > result[0][0] and result[2][0] > result[1][0]:
            ani = "owl"

        print("\nInput image is the {} image number {}".format(animal, img_nb-100))
        print("This image is an " + ani + "\n")

main()
