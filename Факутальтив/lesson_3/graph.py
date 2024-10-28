import numpy as np
import matplotlib.pyplot as plt

x = []
v = []
with open('output.txt') as file:
    for line in file:
        values = line.strip().split(' ')
        x.append(float(values[0]))
        v.append(float(values[1]))

# x1 = []
# v1 = []
# with open('output_Heun.txt') as file:
#     for line in file:
#         values1 = line.strip().split(' ')
#         x1.append(float(values1[0]))
#         v1.append(float(values1[1]))
#
# x2 = []
# v2 = []
# with open('output_R_K.txt') as file:
#     for line in file:
#         values2 = line.strip().split(' ')
#         x2.append(float(values2[0]))
#         v2.append(float(values2[1]))

t = np.arange(0, 20, 0.1)

fig, axs = plt.subplots(1, 3)
axs[0].plot(t, v)
axs[0].plot(t, x)
axs[0].grid()

# axs[1].plot(t, v1)
# axs[1].plot(t, x1)
# axs[1].grid()
#
# axs[2].plot(t, v2)
# axs[2].plot(t, x2)
# axs[2].grid()

plt.show()