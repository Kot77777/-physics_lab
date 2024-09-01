import numpy as np

# P = np.array([7.006, 9.463, 11.921, 14.38, 16.833])
# dl = np.array([0.13, 0.15, 0.16, 0.173, 0.185])
P = np.array([4.618, 9.254, 14.284, 19.284, 24.371])
dl = np.array([0.109, 0.22, 0.34, 0.465, 0.592])

sr_P = sum(P)/5
sr_dl = sum(dl)/5
sr_Pdl = sum(P*dl)/5
sr_kv_dl = sum(dl**2)/5
sr_dl_kv = sr_dl**2
sr_kv_P = sum(P**2)/5
sr_P_kv = sr_P**2

k = (sr_Pdl - sr_P*sr_dl)/(sr_kv_dl - sr_dl_kv)
print(k)

sigma_sluch_k = (1/(3**0.5))*(((sr_kv_P - sr_P_kv)/(sr_kv_dl - sr_dl_kv) - k**2)**0.5)
print(sigma_sluch_k)

a_1 = np.array([2.05, 2, 2.03, 2, 2.05])
b_1 = np.array([1.05, 1.05, 1.03, 1.03, 1.05])
print(sum(a_1)/5, sum(b_1)/5)
a_2 = np.array([2.13, 2.15, 2.15, 2.15, 2.16])
b_2 = np.array([0.4, 0.43, 0.4, 0.4, 0.4])
print(sum(a_2)/5, sum(b_2)/5)

sigma_sluch_a_1 = ((sum(np.array([i - 2 for i in a_1])))/(20))**0.5
sigma_sluch_b_1 = ((sum(np.array([i - 1.04 for i in b_1])))/(20))**0.5
sigma_sluch_a_2 = ((sum(np.array([i - 2.1 for i in a_2])))/(20))**0.5
sigma_sluch_b_2 = ((sum(np.array([i - 0.4 for i in b_2])))/(20))**0.5
print(sigma_sluch_a_1, sigma_sluch_b_1)
print(sigma_sluch_a_2, sigma_sluch_b_2)