import numpy as np

class MutlilayerPerceptrion:
    def __init__(self, nb_inputs, nb_hidden, nb_outputs):
        self.input_nodes = nb_inputs
        self.hidden_nodes = nb_hidden
        self.output_nodes = nb_outputs

        self.weights_ih = np.matrix(hidden_nodes, input_nodes)
        self.weights_ho = np.matrix(output_nodes, hidden_nodes)

    def feedforward(self, inputs):
        pass


def main():
    brain = MutlilayerPerceptrion(2, 2, 1)
    inputs = [1, 0]
    output = brain.feedforward(inputs)

main()
