import random
from matrix import Matrix
from multilayer_perceptron import MutlilayerPerceptron

xor_data = [ [[1], [0]], [[0], [1]], [[0], [0]], [[1], [1]] ]
correct_outputs = [ [[1]], [[1]], [[0]], [[0]] ]

layers = [2, 1, 2, 1]
learning_rate = 0.01
neural_net = MutlilayerPerceptron(layers, learning_rate)

test = Matrix()
test.give_values([[0], [0]])

for i in range(5000):
    input_rand = random.choice(xor_data)
    output_rand = correct_outputs[xor_data.index(input_rand)]
    neural_net.train(input_rand, output_rand)
print(neural_net.feedforward(test)[-1].values)
