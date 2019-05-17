from matrix import Matrix

class MutlilayerPerceptrion:
    def __init__(self, nb_inputs, nb_hidden, nb_outputs, learning_rate=0.1):
        self.learning_rate = learning_rate
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
        return output, hidden

    # Error = answers - guess
    def train(self, inputs, answers):
        guess, hidden = self.feedforward(inputs)
        guess = guess.multiply_scalar(-1)
        #print(guess.values)
        output_errors = answers.add_matrices(guess)
        #print(output_errors.values)

        gradients = guess.apply_derivative_sigmoid()
        gradients = gradients.multiply_matrices(output_errors)
        gradients = gradients.multiply_scalar(self.learning_rate)
        #print(gradients.values)

        hidden_T = hidden.transpose()
        weights_ho_deltas = gradients.multiply_matrices(hidden_T)
        self.weights_ho = self.weights_ho.add_matrices(weights_ho_deltas)
        self.bias_o = self.bias_o.add_matrices(gradients)

        who_t = self.weights_ho.transpose()
        hidden_errors = who_t.multiply_matrices(output_errors)
        print(hidden_errors.values)

        h_gradients = hidden.apply_derivative_sigmoid()
        print(h_gradients.values)
        h_gradients = h_gradients.multiply_matrices(hidden_errors)
        h_gradients = h_gradients.multiply_scalar(self.learning_rate)
        #print(h_gradients.values)
'''
        inputs_T = inputs.transpose()
        weights_ih_deltas = h_gradients.multiply_matrices(inputs_T)
        self.weights_ih = self.weights_ih.add_matrices(weights_ih_deltas)
        self.bias_h = self.bias_h.add_matrices(h_gradients)
'''

def main():
    data = [[2], [0]]
    brain = MutlilayerPerceptrion(2, 2, 1)
    inputs = Matrix()
    answers = Matrix()
    answers.give_values([[1]])
    inputs.give_values(data)
    brain.train(inputs, answers)

main()
