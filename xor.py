import random
from matrix import Matrix
from multilayer_perceptron import MutlilayerPerceptron

xor_data = [ [[1], [0]], [[0], [1]], [[0], [0]], [[1], [1]] ]
correct_outputs = [ [[1]], [[1]], [[0]], [[0]] ]

layers = [2, 1, 2, 1]
learning_rate = 0.1
neural_net = MutlilayerPerceptron(layers, learning_rate)

for i in range(10000):
    input_rand = random.choice(xor_data)
    output_rand = correct_outputs[xor_data.index(input_rand)]
    neural_net.train(input_rand, output_rand)

for i in range(len(xor_data)):
    inputs = Matrix(2, 1)
    inputs.give_values(xor_data[i])
    print(inputs.values)
    print(neural_net.feedforward(inputs)[-1].values)
    print()
print()
print([neural_net.weight_matrices[i].values for i in range(len(neural_net.weight_matrices))])
