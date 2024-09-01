import numpy as np

beta = np.array([0.1885, 0.318, 0.437, 0.669])
M = np.array([0.000357, 0.000492, 0.000649, 0.0001012])

sr_beta_M = sum(beta*M)/4
sr_beta = sum(beta)/4
sr_M = sum(M)/4
sr_M_kv = sum(M**2)/4
sr_kv_M = sr_M**2
sr_beta_kv = sum(beta**2)/4
sr_kv_beta = sr_beta**2

b = (sr_beta_M - sr_beta*sr_M)/(sr_M_kv - sr_kv_M)
a = sr_beta - b*sr_M
print(a, b)

sigma_b = (1/(4**0.5))*((sr_beta_kv - sr_kv_beta)/(sr_M_kv - sr_kv_M) - b**2)**0.5
sigma_a = sigma_b*((sr_M_kv - sr_kv_M)**0.5)
print(sigma_a, sigma_b)

I = np.array([0.000099, 0.000208, 0.000249])
R_kv = (np.array([0.0559, 0.1218, 0.1466]))**2

sr_I_R = sum(I*R_kv)/3
sr_I = sum(I)/3
sr_R = sum(R_kv)/3
sr_R_kv = sum(R_kv**2)/3
sr_kv_R = sr_R**2
sr_I_kv = sum(I**2)/3
sr_kv_I = sr_I**2

b_1 = (sr_I_R - sr_I*sr_R)/(sr_R_kv - sr_kv_R)
a_1 = sr_I - b_1*sr_R
print(a_1, b_1)

sigma_b_1 = (1/(3**0.5))*(((sr_I_kv - sr_kv_I)/(sr_R_kv - sr_kv_R) - b_1**2)**0.5)
sigma_a_1 = sigma_b_1*(sr_R_kv - sr_kv_R)
print(sigma_a_1, sigma_b_1)