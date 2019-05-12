from math import exp
from matrix import Matrix

class MutlilayerPerceptrion:
    def __init__(self, nb_inputs, nb_hidden, nb_outputs):
        self.input_nodes = nb_inputs
        self.hidden_nodes = nb_hidden
        self.output_nodes = nb_outputs

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize(-1, 1)
        self.weights_ho.randomize(-1, 1)

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomize(-1, 1)
        self.bias_o.randomize(-1, 1)

    def feedforward(self, inputs):
        hidden = self.weights_ih.multiply_matrices(inputs)
        hidden = hidden.add_vector(self.bias_h)
        hidden.add_scalar(self.sigmoid(1))

        output = self.weights_ho.multiply_matrices(hidden)
        output = output.add_vector(self.bias_o)
        output.add_scalar(self.sigmoid(1))
        return output

    def train(inputs, answers):
        pass

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

def main():
    data = [[2, 0], [1, 2]]
    brain = MutlilayerPerceptrion(2, 2, 1)
    inputs = Matrix()
    inputs.give_values(data)
    output = brain.feedforward(inputs).values
    print(output)

main()
