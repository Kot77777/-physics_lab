import matplotlib.pyplot as plt
import numpy as np


def func (param):
    x = [param[0]*np.sin(param[1]*t + param[4]) for t in np.arange(0., 2*np.pi, 0.01)]
    y = [param[2]*np.sin(param[3]*t) for t in np.arange(0., 2*np.pi, 0.01)]

    return [x, y]

parametrs = [[1, 1, 1, 2, np.pi/2],
             [1, 3, 1, 2, np.pi/2],
             [1, 3, 1, 4, np.pi/2],
             [1, 5, 1, 4, np.pi/2],
             [1, 5, 1, 6, np.pi/2],
             [1, 9, 1, 8, np.pi/2]]

plt.suptitle('Фигуры Лиссажу')

for i in range(1, 7):
    x = np.array(func(parametrs[i-1])[0])
    y = np.array(func(parametrs[i-1])[1])
    plt.subplot(2, 3, i)
    plt.scatter(x, y)
    plt.title(f'Plot {i}: a = {parametrs[i-1][1]}, b = {parametrs[i-1][3]}')


plt.show()
