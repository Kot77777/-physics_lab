import numpy as np
import matplotlib.pyplot as plt

T = np.array([3.93, 3.32, 2.78, 2.26, 1.72, 1.18])
P = np.array([4, 3.5, 3, 2.5, 2, 1.5])

T1 = np.array([3.57, 2.99, 2.41, 1.88, 1.47, 0.96])
P1 = np.array([4, 3.5, 3, 2.5, 2, 1.5])

T2 = np.array([2.5, 1.81, 1.39, 1, 0.76, 0.49])
P2 = np.array([4, 3.5, 3, 2.5, 2, 1.5])

t = np.polyfit(P, T,  1)
f = np.poly1d(t)
print(f)

t1 = np.polyfit(P1, T1,  1)
f1 = np.poly1d(t1)
print(f1)

t2 = np.polyfit(P2, T2,  1)
f2 = np.poly1d(t2)
print(f2)

fig, ax = plt.subplots()

plt.scatter(P, T, c="g", zorder = 1,)
plt.plot(P, f(P),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0, label = 'T = 21.2℃')

plt.scatter(P1, T1, c="orange", zorder = 1,)
plt.plot(P1, f1(P1),
         linestyle='-', alpha = 1,
         color = "orange", ms=3, zorder = 0, label = 'T = 30℃')

plt.scatter(P2, T2, c="b", zorder = 1,)
plt.plot(P2, f2(P2),
         linestyle='-', alpha = 1,
         color = "b", ms=3, zorder = 0, label = 'T = 50℃')

plt.grid(linewidth = 0.5)

plt.ylabel('dT, ℃', fontsize=10)
plt.xlabel('dP, атм', fontsize=10)
plt.legend()
plt.savefig("График зависимости dT от dP", dpi=600)

plt.show()