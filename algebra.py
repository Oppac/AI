import numpy as np

class Matrix:
    def __init__(self, nb_rows, nb_cols, zero=False):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        if zero:
            self.values = [[0 for i in range(self.nb_cols)] for _ in range(self.nb_rows)]
        else:
            self.values = [[np.random.randint(-10, 10)
                          for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    def add_scalar(self, scalar):
        self.values = [[j + scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]

    def multiply_scalar(self, scalar):
        self.values = [[j * scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]


    def add_matrix(self, matrix):
        if (self.nb_rows == matrix.nb_rows) and (self.nb_cols == matrix.nb_cols):
            self.values = [[j + k for j, k in zip(self.values[i], matrix.values[i])]
                          for i in range(self.nb_rows)]
        else:
            print("Matrices must have the same size")


####################################################


def add_matrices(m1, m2):
    if (m1.nb_rows == m2.nb_rows) and (m1.nb_cols == m2.nb_cols):
        m3 = Matrix(m1.nb_rows, m1.nb_cols, True)
        m3.values = [[j + k for j, k in zip(m1.values[i], m2.values[i])]
                     for i in range(m1.nb_rows)]
        return m3
    else:
        print("Matrices must have the same size")

def main():
    m = Matrix(2, 3)
    print(m.values)
    m2 = Matrix(2, 3)
    print(m2.values)
    m3 = add_matrices(m, m2)
    print(m3.values)

main()
