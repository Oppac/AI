import random
from matrix import Matrix
from multilayer_perceptron import MutlilayerPerceptron

neural_net = MutlilayerPerceptron(2, 2, 1)

input_data = [ [[1], [0]], [[0], [1]], [[0], [0]], [[1], [1]] ]
correct_outputs = [ [[1]], [[1]], [[0]], [[0]] ]

test = Matrix()
test.give_values([[1], [0]])

for i in range(5000):
    input_rand = random.choice(input_data)
    output_rand = correct_outputs[input_data.index(input_rand)]
    neural_net.train(input_rand, output_rand)
print(neural_net.feedforward(test)[0].values)
