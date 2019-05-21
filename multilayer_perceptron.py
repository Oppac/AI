from matrix import Matrix

class MutlilayerPerceptron:
    "Layers: [nb_input_nodes, nb_output_nodes, nb_hidden_nodes, nb_hidden_layers]"
    def __init__(self, layers, learning_rate=0.1):
        if len(layers) == 4:
            self.learning_rate = learning_rate
            self.input_nodes = layers[0]
            self.output_nodes = layers[1]
            self.hidden_nodes = layers[2]
            self.hidden_layers = layers[3]
            self.weight_matrices = self.weights_init()
            self.bias_vectors = self.bias_init()
        else:
            exit("Invalid layer size: [nb_input_nodes, nb_output_nodes" +
            "nb_hidden_nodes, nb_hidden_layers]")

    def new_matrix(self, nb_rows, nb_cols):
        w_matrix = Matrix(nb_rows, nb_cols)
        w_matrix.randomize(-1, 1)
        return w_matrix

    def weights_init(self):
        weight_matrices = []

        w_input_hidden = self.new_matrix(self.hidden_nodes, self.input_nodes)
        weight_matrices.append(w_input_hidden)

        for _ in range(self.hidden_layers-1):
            w_hidden_hidden = self.new_matrix(self.hidden_nodes, self.hidden_nodes)
            weight_matrices.append(w_hidden_hidden)

        w_hidden_output = self.new_matrix(self.output_nodes, self.hidden_nodes)
        weight_matrices.append(w_hidden_output)

        return weight_matrices

    def bias_init(self):
        bias_vectors = []
        for _ in range(self.hidden_layers):
            hidden_bias = self.new_matrix(self.hidden_nodes, 1)
            bias_vectors.append(hidden_bias)

        output_bias = self.new_matrix(self.output_nodes, 1)
        bias_vectors.append(output_bias)
        return bias_vectors

################################################################################

    def feedforward(self, inputs):
        hidden_guess = self.weight_matrices[0].multiply_matrices(inputs)
        hidden_guess = hidden_guess.add_vector(self.bias_vectors[0])
        hidden_guess = hidden_guess.apply_sigmoid()

        output_guess = self.weight_matrices[-1].multiply_matrices(hidden_guess)
        output_guess = output_guess.add_vector(self.bias_vectors[-1])
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
        w_hidden_output_T = self.weight_matrices[-1].transpose()
        hidden_errors = w_hidden_output_T.multiply_matrices(output_errors)

        gradient_ouput = self.get_gradient(output_guess, output_errors)
        self.bias_vectors[-1] = self.bias_vectors[-1].add_matrices(gradient_ouput)
        gradient_hidden = self.get_gradient(hidden_guess, hidden_errors)
        self.bias_vectors[0] = self.bias_vectors[0].add_matrices(gradient_hidden)

        hidden_T = hidden_guess.transpose()
        deltaW_output = gradient_ouput.multiply_matrices(hidden_T)
        self.weight_matrices[-1] = self.weight_matrices[-1].add_matrices(deltaW_output)

        inputs_T = inputs.transpose()
        deltaW_hidden = gradient_hidden.multiply_matrices(inputs_T)
        self.weight_matrices[0] = self.weight_matrices[0].add_matrices(deltaW_hidden)


    def train(self, input_data, correct_outputs):
        inputs = Matrix(); answers = Matrix()
        inputs.give_values(input_data)
        answers.give_values(correct_outputs)

        output_guess, hidden_guess = self.feedforward(inputs)
        output_errors = self.guess_error(output_guess, answers)
        self.backpropagation(output_guess, output_errors, hidden_guess, inputs)
