# Toy matrix class for implementing the neural network from scratch

from math import exp
from random import uniform

class Matrix:
    # Matrix of size [rows * columns]
    def __init__(self, nb_rows=1, nb_cols=1, values=None):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.values = [[0 for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    # Load values into the matrix
    def give_values(self, values):
        self.nb_rows = len(values)
        try:
            self.nb_cols = len(values[0])
        except:
            self.nb_cols = 1
        self.values = values

    # Given random values to the matrix within the bounds
    def randomize(self, lower, upper):
        self.values = [[uniform(lower, upper)
                      for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    # Add a scalar to the values of the matrix
    def add_scalar(self, scalar):
        result = Matrix(self.nb_rows, self.nb_cols)
        result.values = [[j + scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]
        return result

    # Multiply all values by a scalar
    def multiply_scalar(self, scalar):
        result = Matrix(self.nb_rows, self.nb_cols)
        result.values = [[j * scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]
        return result

    # Add a the elements of the vector to the matrix
    def add_vector(self, vector):
        m3 = Matrix(self.nb_rows, self.nb_cols)
        if (vector.nb_cols == 1):
            m3.values = [[j + vector.values[i][0] for j in self.values[i]]
                         for i in range(self.nb_rows)]
        else:
            exit("{}x{} is not a vector".format(vector.nb_rows, vector.nb_cols))
        return m3

    # Hadamard product with a vector
    def multiply_vector(self, vector):
        m3 = Matrix(self.nb_rows, self.nb_cols)
        if (vector.nb_cols == 1):
            m3.values = [[j * vector.values[i][0] for j in self.values[i]]
                         for i in range(self.nb_rows)]
        else:
            exit("{}x{} is not a vector".format(vector.nb_rows, vector.nb_cols))
        return m3

    # Matrices addition
    def add_matrices(self, m2):
        m3 = Matrix(self.nb_rows, self.nb_cols)
        if (self.nb_rows == m2.nb_rows) and (self.nb_cols == m2.nb_cols):
            m3.values = [[j + k for j, k in zip(self.values[i], m2.values[i])]
                         for i in range(self.nb_rows)]
        else:
            exit("Invalid size addition: {}x{} != {}x{}".format(
                  self.nb_rows, self.nb_cols, m2.nb_rows, m2.nb_cols))
        return m3

    # Dot product
    def multiply_matrices(self, m2):
        m3 = Matrix(self.nb_rows, m2.nb_cols)
        if (self.nb_cols == m2.nb_rows):
            for i in range(self.nb_rows):
                for j in range(m2.nb_cols):
                    for k in range(m2.nb_rows):
                        m3.values[i][j] += self.values[i][k] * m2.values[k][j]
        else:
            print(self.nb_rows, self.nb_cols)
            exit("Invalid size multiply: {} != {}".format(self.nb_cols, m2.nb_rows))
        return m3

    # Matrix transpose
    def transpose(self):
        trans = Matrix(self.nb_cols, self.nb_rows)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                trans.values[j][i] += self.values[i][j]
        return trans

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    # ReLu activation function
    def relu(self, x):
        return max(x, 0)

    def relu_prime(self, x):
        return float(x > 0)

    # Apply the sigmoid function to the elements of the matrix
    def apply_sigmoid(self):
        result = Matrix(self.nb_rows, self.nb_cols)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                result.values[i][j] += self.sigmoid(self.values[i][j])
        return result

    # Apply the derivative of the sigmoid function to the matrix
    def apply_sigmoid_prime(self):
        result = Matrix(self.nb_rows, self.nb_cols)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                result.values[i][j] += (self.sigmoid(self.values[i][j])
                                       * (1.0 - self.sigmoid(self.values[i][j])))
        return result

    # Apply the reLu function to the elements of the matrix
    def apply_relu(self):
        result = Matrix(self.nb_rows, self.nb_cols)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                result.values[i][j] += self.relu(self.values[i][j])
        return result

    # Apply the derivative of the reLu function to the matrix
    def apply_relu_prime(self):
        result = Matrix(self.nb_rows, self.nb_cols)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                result.values[i][j] += self.relu_prime(self.values[i][j])
        return result

    def print_size(self):
        print("Size: {}x{}".format(self.nb_rows, self.nb_cols))
