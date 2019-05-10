import sys
import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(x_data, y_data, learning_rate):
    m = 0; b = 0
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
        guess = m * x + b
        error = y - guess
        m += (error * x) * learning_rate
        b += error * learning_rate
    return m, b

def new_plot(x_data, y_data, plt):
    plt.clf()
    plt.scatter(x=x_data, y=y_data, s=30, marker='o', edgecolors="black")
    m, b = gradient_descent(x_data, y_data, 0.01)
    plt.plot(x_data, m*x_data+b, '-r')
    plt.draw()

def show(x_data, y_data):
    fig, ax = plt.subplots()
    new_plot(x_data, y_data, plt)
    plt.show()

def main():
    global x_data, y_data
    size = 100
    learning_rate = 0.0001

    x_data = np.random.normal(6, 3, size)
    y_data = np.random.normal(6, 3, size)
    show(x_data, y_data)


main()
