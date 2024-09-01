import numpy as np
import matplotlib.pyplot as plt

M = np.array([40.898, 32.428, 26.015, 20.933, 17.182, 0])
Omeg = np.array([20.93, 16.69, 13.43, 10.89, 8.84, 0])

fig, ax = plt.subplots()

t1 = np.polyfit(M, Omeg, 1)
f1 = np.poly1d(t1)

plt.plot(M, f1(M), '*', label = 'k = 0.514 \u00B1 0.03',
         linestyle='-', mfc = 'r',mec = 'r', alpha = 1,
         color = 'blue', ms=5)
plt.errorbar(M, f1(M), xerr=0, yerr=0.4, fmt='|', color='black')
plt.xlim([0, 42])
plt.ylim([0, 22])
plt.grid(linewidth = 0.5)
plt.xlabel('M*10^(-2), Н*м', fontsize=10)
plt.ylabel('\u03A9*10^(-2), с^(-1)', fontsize=10)

plt.legend()
plt.show()