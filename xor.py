import random
from matrix import Matrix
from multilayer_perceptron import MutlilayerPerceptron

xor_data = [ [[1], [0]], [[0], [1]], [[0], [0]], [[1], [1]] ]
labels = [ [[1]], [[1]], [[0]], [[0]] ]

def create_training_data(sample_size):
    training_set = []

    for i in range(sample_size):
        data_matrix = Matrix(); label_matrix = Matrix()

        data = random.choice(xor_data)
        label = labels[xor_data.index(data)]

        data_matrix.give_values(data)
        label_matrix.give_values(label)

        new_data = [data_matrix, label_matrix]
        training_set.append(new_data)

    return training_set

def main():
    layers = [2, 2, 1, 1]
    learning_rate = 0.1
    sample_size = 1000
    batch_size = 10
    epochs = 4
    neural_net = MutlilayerPerceptron(layers, learning_rate, batch_size, epochs)

    training_set = create_training_data(sample_size)
    neural_net.train(training_set)

    print()
    for i in range(len(xor_data)):
        inputs = Matrix(2, 1)
        inputs.give_values(xor_data[i])
        print(inputs.values)
        print(neural_net.feedforward(inputs)[-1].values)
        print()


main()
