# importing libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.histograms import histogram

# generating two series of random values
# using numpy random module of shape (500,1)
series1 = np.random.randn(500, 1)
series2 = np.random.randn(400, 1)

print(series2)
# plotting first histogram
plt.hist(series1, label='series1', alpha=.8, edgecolor='red')

# plotting second histogram
plt.hist(series2, label='series2', alpha=0.7, edgecolor='yellow')
plt.legend()

# Showing the plot using plt.show()
plt.show()
