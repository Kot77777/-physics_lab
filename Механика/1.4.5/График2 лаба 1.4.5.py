import numpy as np
import matplotlib.pyplot as plt

U = np.array([0, 135.9, 154.4, 179.6, 200.9, 220.5])
U2 = U*U
T = np.array([0, 10.27, 13.5, 18.3, 23, 27.7])

size = [10, 10, 10, 10, 10, 10]
plt.scatter(T, U2, size, c="black")
t6 = np.polyfit(T, U2 , 1)
f6 = np.poly1d(t6)
plt.plot(T, f6(T), label = 'k = 1759,8 \u00B1 26', alpha=0.5)

plt.xlabel('T', fontsize=10, fontweight='bold')
plt.ylabel('U^2, (м/с)^2', fontsize=10, fontweight='bold')

plt.grid(linewidth = 0.5)
plt.legend()
plt.show()
