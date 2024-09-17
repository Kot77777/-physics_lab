import numpy as np
n = int(input())

matrix_A = 2*np.eye(n, dtype=int)
for i in range(n):
    for j in range(n):
        if (j == i + 1) or (j == i - 1):
            matrix_A[i][j] = -1

b = np.zeros(n, dtype=int)
b[0] = 100
if(n!=1):b[-1] = 1

solution = np.linalg.solve(matrix_A, b)

print(solution)