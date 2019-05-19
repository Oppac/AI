from matrix import Matrix

class MutlilayerPerceptron:
    def __init__(self, nb_inputs, nb_hidden, nb_outputs, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.input_nodes = nb_inputs
        self.hidden_nodes = nb_hidden
        self.output_nodes = nb_outputs
        self.weights_init()

    def weights_init(self):
        self.w_input_hidden = Matrix(self.hidden_nodes, self.input_nodes)
        self.w_input_hidden.randomize(-1, 1)
        self.w_hidden_output = Matrix(self.output_nodes, self.hidden_nodes)
        self.w_hidden_output.randomize(-1, 1)

        self.bias_hidden = Matrix(self.hidden_nodes, 1)
        self.bias_hidden.randomize(-1, 1)
        self.bias_output = Matrix(self.output_nodes, 1)
        self.bias_output.randomize(-1, 1)

    def feedforward(self, inputs):
        hidden_guess = self.w_input_hidden.multiply_matrices(inputs)
        hidden_guess = hidden_guess.add_vector(self.bias_hidden)
        hidden_guess = hidden_guess.apply_sigmoid()

        output_guess = self.w_hidden_output.multiply_matrices(hidden_guess)
        output_guess = output_guess.add_vector(self.bias_output)
        output_guess = output_guess.apply_sigmoid()
        return output_guess, hidden_guess

    # Error = answers - guess
    def guess_error(self, guess, answers):
        guess = guess.multiply_scalar(-1)
        errors = answers.add_matrices(guess)
        return errors

    def get_gradient(self, guess, error):
        gradient = guess.apply_derivative_sigmoid()
        gradient = gradient.multiply_vector(error)
        gradient = gradient.multiply_scalar(self.learning_rate)
        return gradient

    def backpropagation(self, output_guess, output_errors, hidden_guess, inputs):
        w_hidden_output_T = self.w_hidden_output.transpose()
        hidden_errors = w_hidden_output_T.multiply_matrices(output_errors)

        gradient_ouput = self.get_gradient(output_guess, output_errors)
        self.bias_output = self.bias_output.add_matrices(gradient_ouput)
        gradient_hidden = self.get_gradient(hidden_guess, hidden_errors)
        self.bias_hidden = self.bias_hidden.add_matrices(gradient_hidden)

        hidden_T = hidden_guess.transpose()
        deltaW_output = gradient_ouput.multiply_matrices(hidden_T)
        self.w_hidden_output = self.w_hidden_output.add_matrices(deltaW_output)

        inputs_T = inputs.transpose()
        deltaW_hidden = gradient_hidden.multiply_matrices(inputs_T)
        self.w_input_hidden = self.w_input_hidden.add_matrices(deltaW_hidden)


    def train(self, input_data, correct_outputs):
        inputs = Matrix(); answers = Matrix()
        inputs.give_values(input_data)
        answers.give_values(correct_outputs)

        output_guess, hidden_guess = self.feedforward(inputs)
        output_errors = self.guess_error(output_guess, answers)
        self.backpropagation(output_guess, output_errors, hidden_guess, inputs)
