import numpy as np
import matplotlib.pyplot as plt

T = 1 / np.array([294.2, 303, 323])
mu = np.array([1.09, 1.04, 0.78])

t = np.polyfit(T, mu,  1)
f = np.poly1d(t)
print(t)

fig, ax = plt.subplots()

plt.scatter(T, mu, c="g", zorder = 1,)
plt.plot(T, f(T),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0)

plt.grid(linewidth = 0.5)

plt.ylabel('mu, К/атм', fontsize=10)
plt.xlabel('1/T, 1/K', fontsize=10)
plt.savefig("График зависимости mu от T^(-1)", dpi=600)

plt.show()