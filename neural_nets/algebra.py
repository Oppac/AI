from random import randint

class Matrix:
    def __init__(self, nb_rows, nb_cols, values=None):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        if values:
            self.values = values
        else:
            self.values = [[0 for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    def randomize(self, lower, upper):
        self.values = [[randint(lower, upper)
                      for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    def add_scalar(self, scalar):
        self.values = [[j + scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]

    def multiply_scalar(self, scalar):
        self.values = [[j * scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]



####################################################


def add_matrices(m1, m2):
    if (m1.nb_rows == m2.nb_rows) and (m1.nb_cols == m2.nb_cols):
        m3 = Matrix(m1.nb_rows, m1.nb_cols)
        m3.values = [[j + k for j, k in zip(m1.values[i], m2.values[i])]
                     for i in range(m1.nb_rows)]
        return m3
    else:
        print("Matrices must have the same size")


def multiply_matrices(m1, m2):
    val = []
    if (m1.nb_cols == m2.nb_rows):
        m3 = Matrix(m1.nb_rows, m2.nb_cols)
        for i in range(m1.nb_rows):
            for j in range(m2.nb_cols):
                for k in range(m2.nb_rows):
                    m3.values[i][j] += m1.values[i][k] * m2.values[k][j]
        return m3
    else:
        print("Invalid size: m1.nb_cols != m2.nb_rows")

def main():
    d1 = [[1,1,2],[2,1,1]]
    d2 = [[1, 1], [1, 1], [1, 2]]
    m1 = Matrix(2, 3, d1)
    print(m1.values)
    m2 = Matrix(3, 2, d2)
    print(m2.values)
    m3 = multiply_matrices(m1, m2)
    print(m3.values)

main()
