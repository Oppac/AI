import random
from matrix import Matrix
from multilayer_perceptron import MutlilayerPerceptron

xor_data = [ [[1], [0]], [[0], [1]], [[0], [0]], [[1], [1]] ]
correct_outputs = [ [[1]], [[1]], [[0]], [[0]] ]

layers = [2, 2, 1, 1]
learning_rate = 0.1
neural_net = MutlilayerPerceptron(layers, learning_rate)

for i in range(1000):
    input_rand = random.choice(xor_data)
    output_rand = correct_outputs[xor_data.index(input_rand)]
    neural_net.train(input_rand, output_rand)

print()
for i in range(len(xor_data)):
    inputs = Matrix(2, 1)
    inputs.give_values(xor_data[i])
    print(inputs.values)
    print(neural_net.feedforward(inputs)[-1].values)
    print()
