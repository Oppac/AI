from matrix import Matrix

class MutlilayerPerceptrion:
    def __init__(self, nb_inputs, nb_hidden, nb_outputs):
        self.input_nodes = nb_inputs
        self.hidden_nodes = nb_hidden
        self.output_nodes = nb_outputs

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ih.randomize(-1, 1)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ho.randomize(-1, 1)

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_h.randomize(-1, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_o.randomize(-1, 1)

    def feedforward(self, inputs):
        hidden = self.weights_ih.multiply_matrices(inputs)
        hidden = hidden.add_vector(self.bias_h)
        hidden = hidden.apply_sigmoid()

        output = self.weights_ho.multiply_matrices(hidden)
        output = output.add_vector(self.bias_o)
        output = output.apply_sigmoid()
        return output

    # Error = answers - guess
    def train(self, inputs, answers):
        guess = self.feedforward(inputs)
        guess = guess.multiply_scalar(-1)
        print(guess.values)
        final_errors = answers.add_matrices(guess)
        print(final_errors.values)
        who_t = self.weights_ho.transpose()
        print(who_t.values)
        hidden_errors = who_t.multiply_matrices(final_errors)
        print(hidden_errors.values)

        gradients = guess.apply_derivative_sigmoid()
        gradients.multiply_matrices(final_errors)
        print(gradients.values)


def main():
    data = [[2, 0], [1, 2]]
    brain = MutlilayerPerceptrion(2, 2, 1)
    inputs = Matrix()
    answers = Matrix()
    answers.give_values([[1, 1]])
    inputs.give_values(data)
    brain.train(inputs, answers)

main()
