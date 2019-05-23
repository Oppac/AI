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
        bias_vectors.append(Matrix(self.input_nodes, 1))
        for _ in range(self.hidden_layers):
            hidden_bias = self.new_matrix(self.hidden_nodes, 1)
            bias_vectors.append(hidden_bias)

        output_bias = self.new_matrix(self.output_nodes, 1)
        bias_vectors.append(output_bias)
        return bias_vectors

################################################################################

    def feedforward(self, inputs):
        guesses = [inputs]
        for i in range(len(self.weight_matrices)):
            guess = self.weight_matrices[i].multiply_matrices(guesses[i])
            guess = guess.add_vector(self.bias_vectors[i])
            print(guess.values)
            guess = guess.apply_sigmoid()
            guesses.append(guess)
        return guesses

    # Error = answers - guess
    def guess_error(self, guesses, answers):
        guesses = guesses.multiply_scalar(-1)
        errors = answers.add_matrices(guesses)
        return errors

    def get_gradient(self, guess, error):
        gradient = guess.apply_derivative_sigmoid()
        gradient = gradient.multiply_vector(error)
        gradient = gradient.multiply_scalar(self.learning_rate)
        return gradient

    def backpropagation(self, guesses, answers):
        error = self.guess_error(guesses[-1], answers)
        for i in range(self.hidden_layers+1, 0, -1):
            weights_transpose = self.weight_matrices[i-1].transpose()
            delta = self.get_gradient(guesses[i], error)
            self.bias_vectors[i] = self.bias_vectors[i].add_matrices(delta)
            previous_guess_T = guesses[i-1].transpose()
            delta = delta.multiply_matrices(previous_guess_T)
            self.weight_matrices[i-1] = self.weight_matrices[i-1].add_matrices(delta)
            error = weights_transpose.multiply_matrices(error)

    def train(self, input_data, correct_outputs):
        inputs = Matrix(); answers = Matrix()
        inputs.give_values(input_data)
        answers.give_values(correct_outputs)

        guesses = self.feedforward(inputs)
        self.backpropagation(guesses, answers)
