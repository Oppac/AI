# Toy neural network class implemented from scratch

# TODO -> Saving models

from matrix import Matrix

class MutlilayerPerceptron:
    # Define the architeture of the neural network
    # "Layers: [nb_input_nodes, nb_hidden_nodes, nb_nb_hidden_layers, nb_output_nodes]"
    def __init__(self, layers, learning_rate=0.1, batch_size=16, epochs=4):
        if len(layers) == 4:
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.input_nodes = layers[0]
            self.hidden_nodes = layers[1]
            self.nb_hidden_layers = layers[2]
            self.output_nodes = layers[3]
            self.weight_matrices = self.weights_init()
            self.bias_vectors = self.bias_init()
        else:
            exit("Invalid layer size: [nb_input_nodes, nb_hidden_nodes, " +
            "nb_nb_hidden_layers, nb_output_nodes]")

    # Create a new random matrix for the weigths
    # The upper and lower bound may require to be tweak depending on the problem
    def new_matrix(self, nb_rows, nb_cols):
        w_matrix = Matrix(nb_rows, nb_cols)
        w_matrix.randomize(0, 0.1)
        return w_matrix

    # Initialize the weigths matrices with random values
    def weights_init(self):
        weight_matrices = []

        w_input_hidden = self.new_matrix(self.hidden_nodes, self.input_nodes)
        weight_matrices.append(w_input_hidden)

        for _ in range(self.nb_hidden_layers-1):
            w_hidden_hidden = self.new_matrix(self.hidden_nodes, self.hidden_nodes)
            weight_matrices.append(w_hidden_hidden)

        w_hidden_output = self.new_matrix(self.output_nodes, self.hidden_nodes)
        weight_matrices.append(w_hidden_output)

        return weight_matrices

    # Initialize the bias matrices at zero
    def bias_init(self):
        bias_vectors = []
        for _ in range(self.nb_hidden_layers):
            hidden_bias = Matrix(self.hidden_nodes, 1)
            bias_vectors.append(hidden_bias)

        output_bias = Matrix(self.output_nodes, 1)
        bias_vectors.append(output_bias)
        return bias_vectors

################################################################################

    # Warning: ReLu do not work properly
    # prime=True => derivative of the activation function
    def activation(self, guess, prime=False):
        if not prime:
            guess = guess.apply_sigmoid()
            #guess = guess.apply_relu()
        else:
            guess = guess.apply_sigmoid_prime()
            #guess = guess.apply_relu_prime()
        return guess

    # Guess = weigths * layer_inputs + bias
    def feedforward(self, inputs):
        guesses = [inputs]
        for i in range(len(self.weight_matrices)):
            guess = self.weight_matrices[i].multiply_matrices(guesses[i])
            guess = guess.add_vector(self.bias_vectors[i])
            guess = self.activation(guess)
            guesses.append(guess)
        return guesses

    # Error = correct_answers - guess
    def guess_error(self, guesses, answers):
        guesses = guesses.multiply_scalar(-1)
        errors = answers.add_matrices(guesses)
        return errors

    # gradient = learning_rate * error_vector * sigmoid_prime(output_guess)
    def get_gradient(self, guess, error):
        gradient = self.activation(guess, True)
        gradient = gradient.multiply_vector(error)
        gradient = gradient.multiply_scalar(self.learning_rate)
        return gradient

    # new_weigths = old_weights + (gradient * input_transpose)
    # Bias increased by the value of the gradient
    # The error of the "previous layer" is the new error
    def backpropagation(self, guesses, answers):
        error = self.guess_error(guesses[-1], answers)
        for i in range(self.nb_hidden_layers+1, 0, -1):
            weights_transpose = self.weight_matrices[i-1].transpose()
            delta = self.get_gradient(guesses[i], error)
            self.bias_vectors[i-1] = self.bias_vectors[i-1].add_matrices(delta)
            previous_guess_T = guesses[i-1].transpose()
            delta = delta.multiply_matrices(previous_guess_T)
            self.weight_matrices[i-1] = self.weight_matrices[i-1].add_matrices(delta)
            error = weights_transpose.multiply_matrices(error)

    def create_batches(self, training_set):
        batches = []; batch = []; j =0
        for i in range(len(training_set)):
            batch.append(training_set[i])
            j += 1
            if j == 16:
                batches.append(batch)
                batch = []; j = 0
        return batches

    # For each input received:
    # 1) Feedforward make a guess
    # 2) The guess is compared to the correct answer
    # 3) Backpropagation to adjust the weights based on the error
    def train(self, training_set):
        for epoch in range(self.epochs):
            all_batches = self.create_batches(training_set)
            for batch in all_batches:
                for input, label in batch:
                    guesses = self.feedforward(input)
                    self.backpropagation(guesses, label)
