import numpy as np
import matplotlib.pyplot as plt

P = np.array([7.006, 9.463, 11.921, 14.38, 16.833])
dl = np.array([0.13, 0.15, 0.16, 0.173, 0.185])

fig, ax = plt.subplots()

t1 = np.polyfit(dl, P, 1)
f1 = np.poly1d(t1)

plt.scatter(dl, P, c="r", zorder = 1)
plt.plot(dl, f1(dl), label = 'k = 182.24 \u00B1 0.68',
         linestyle='-', alpha = 1,
         color = 'blue', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)
plt.xlabel('\u0394l, мм', fontsize=10)
plt.ylabel('P, Н', fontsize=10)

fig1, ax = plt.subplots()

P_1_up = np.array([4.618, 9.254, 14.284, 19.284, 24.371])
y_1max_up = np.array([0.109, 0.22, 0.34, 0.465, 0.592])
y_1max_down = np.array([0.107, 0.222, 0.342, 0.466, 0.592])

t2 = np.polyfit(y_1max_up, P_1_up, 1)
f2 = np.poly1d(t2)

plt.scatter(y_1max_down, P_1_up, c="b", zorder = 1)
plt.scatter(y_1max_up, P_1_up, c="green", zorder = 1)
plt.plot(y_1max_up, f2(y_1max_up), label = 'k = 40.89 \u00B1 3.46',
         linestyle='-', alpha = 1,
         color = 'blue', ms=5, zorder = 0)
plt.plot(y_1max_down, f2(y_1max_up), label = 'k = 40.89 \u00B1 3.46',
         linestyle='-', alpha = 1,
         color = 'green', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)
plt.xlabel('y_max, cм', fontsize=10)
plt.ylabel('P, Н', fontsize=10)

plt.legend()
plt.show()