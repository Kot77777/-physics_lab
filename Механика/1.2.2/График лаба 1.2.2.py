import numpy as np
import matplotlib.pyplot as plt

I = np.array([0.000099, 0.000208, 0.000249])
R_kv = (np.array([0.0559, 0.1218, 0.1466]))**2

fig, ax = plt.subplots()

t1 = np.polyfit(R_kv, I, 1)
f1 = np.poly1d(t1)

plt.scatter(R_kv, I, c="r")
plt.plot(R_kv, f1(R_kv), label = 'I = 0.535R^2 + 0.0038',
         linestyle='-', mfc = 'r', alpha = 1,
         color = 'blue', ms=5)
plt.grid(linewidth = 0.5)
plt.xlabel('R^2, м^2', fontsize=10)
plt.ylabel('I, кг*м^2', fontsize=10)

plt.legend()

beta = np.array([0.1885, 0.318, 0.437, 0.669])
M = np.array([0.000357, 0.000492, 0.000649, 0.0001012])

fig1, ax = plt.subplots()

t1 = np.polyfit(M, beta, 1)
f1 = np.poly1d(t1)

plt.scatter(M, beta, c="r")
plt.plot(M, f1(M), label = '\u03B2_0 = 66.46M - 0.14',
         linestyle='-', mfc = 'r', alpha = 1,
         color = 'blue', ms=5)
plt.grid(linewidth = 0.5)
plt.xlabel('M, Н*м', fontsize=10)
plt.ylabel('\u03B2_0, рад/c^2', fontsize=10)

plt.legend()
plt.show()

