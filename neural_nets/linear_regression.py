import matplotlib.pyplot as plt
import numpy as np

def linear_regression(x_data, y_data):
    size = len(x_data)
    x_sum = 0; y_sum = 0
    num = 0; den = 0

    for i in range(size):
        x_sum += x_data[i]
        y_sum += y_data[i]
    x_mean = x_sum/size; y_mean = y_sum/size

    for i in range(size):
        num += (x_data[i] - x_mean) * (y_data[i] - y_mean)
        den += (x_data[i] - x_mean) * (x_data[i] - x_mean)

    m = num / den
    b = y_mean - m * x_mean
    return m, b

def onclick(event):
    global x_data, y_data
    x_data = np.append(x_data, event.xdata)
    y_data = np.append(y_data, event.ydata)
    m, b = linear_regression(x_data, y_data)
    plt.clf()
    plt.scatter(x=x_data, y=y_data, s=25, marker='o', edgecolors="black")
    plt.plot(x_data, m*x_data+b, '-r')
    plt.draw()

def show(x_data, y_data):
    fig, ax = plt.subplots()
    ax.scatter(x=x_data, y=y_data, s=25, marker='o', edgecolors="black")

    m, b = linear_regression(x_data, y_data)
    plt.plot(x_data, m*x_data+b, '-r')

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def main():
    global x_data, y_data
    size = 2
    x_data = np.random.normal(6, 3, size)
    y_data = np.random.logistic(20, 4, size)
    show(x_data, y_data)



main()
