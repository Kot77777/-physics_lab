import numpy as np
n = np.array(range(1, 9))

nu1 = np.array([i*135.7 for i in range(1, 9)])
nu2 = np.array([i*155.7 for i in range(1, 9)])
nu3 = np.array([i*181.2 for i in range(1, 9)])
nu4 = np.array([i*203.2 for i in range(1, 9)])
nu5 = np.array([i*223 for i in range(1, 9)])
nu6 = np.array([i*241.4 for i in range(1, 9)])

nu_T1 = np.array([134.6, 270, 404.4, 541, 677.3, 815, 951.2, 1092])
nu_T2 = np.array([154, 306, 462, 614, 772, 925, 1083, 1237])
nu_T3 = np.array([178.6, 358, 537, 718, 898, 1078, 1257, 1439])
nu_T4 = np.array([200.2, 400, 601, 801, 1003, 1205, 1407, 1610])
nu_T5 = np.array([219.5, 440, 659, 882, 1100, 1325, 1541, 1768])
nu_T6 = np.array([237.7, 476, 713, 953, 1190, 1429, 1669, 1908])

nu_T1n = nu_T1*n
nu_T2n = nu_T2*n
nu_T3n = nu_T3*n
nu_T4n = nu_T4*n
nu_T5n = nu_T5*n
nu_T6n = nu_T6*n

nn = sum(n*n)/8
nu_T1kv = sum(nu_T1*nu_T1)/8
nu_T2kv = sum(nu_T2*nu_T2)/8
nu_T3kv = sum(nu_T3*nu_T3)/8
nu_T4kv = sum(nu_T4*nu_T4)/8
nu_T5kv = sum(nu_T5*nu_T5)/8
nu_T6kv = sum(nu_T6*nu_T6)/8

print(sum(nu_T1n)/(8*nn), sum(nu_T2n)/(8*nn), sum(nu_T3n)/(8*nn), sum(nu_T4n)/(8*nn), sum(nu_T5n)/(8*nn), sum(nu_T6n)/(8*nn))

delta1 = sum(abs(nu_T1 - nu1))/8
delta2 = sum(abs(nu_T2 - nu2))/8
delta3 = sum(abs(nu_T3 - nu3))/8
delta4 = sum(abs(nu_T4 - nu4))/8
delta5 = sum(abs(nu_T5 - nu5))/8
delta6 = sum(abs(nu_T6 - nu6))/8

delta_sluch1 = (((nu_T1kv/nn) - (sum(nu_T1n)/(8*nn))**2)/7)**0.5
delta_sluch2 = (((nu_T2kv/nn) - (sum(nu_T2n)/(8*nn))**2)/7)**0.5
delta_sluch3 = (((nu_T3kv/nn) - (sum(nu_T3n)/(8*nn))**2)/7)**0.5
delta_sluch4 = (((nu_T4kv/nn) - (sum(nu_T4n)/(8*nn))**2)/7)**0.5
delta_sluch5 = (((nu_T5kv/nn) - (sum(nu_T5n)/(8*nn))**2)/7)**0.5
delta_sluch6 = (((nu_T6kv/nn) - (sum(nu_T6n)/(8*nn))**2)/7)**0.5

delta_sist1 = (sum(nu_T1n)/(8*nn))*(delta1/1092)
delta_sist2 = (sum(nu_T2n)/(8*nn))*(delta2/1237)
delta_sist3 = (sum(nu_T3n)/(8*nn))*(delta3/1439)
delta_sist4 = (sum(nu_T4n)/(8*nn))*(delta4/1610)
delta_sist5 = (sum(nu_T5n)/(8*nn))*(delta5/1768)
delta_sist6 = (sum(nu_T6n)/(8*nn))*(delta5/1908)

sigma1 = (delta_sist1**2 + delta_sluch1**2)**0.5
sigma2 = (delta_sist2**2 + delta_sluch2**2)**0.5
sigma3 = (delta_sist3**2 + delta_sluch3**2)**0.5
sigma4 = (delta_sist4**2 + delta_sluch4**2)**0.5
sigma5 = (delta_sist5**2 + delta_sluch5**2)**0.5
sigma6 = (delta_sist6**2 + delta_sluch6**2)**0.5

U_teor = np.array([135.7, 155.7, 181.2, 203.2, 223, 241.35])
U2_teor = U_teor*U_teor
T = np.array([10.27, 13.5, 18.3, 23, 27.7, 33.11])
U = np.array([sum(nu_T1n)/(8*nn), sum(nu_T2n)/(8*nn), sum(nu_T3n)/(8*nn), sum(nu_T4n)/(8*nn), sum(nu_T5n)/(8*nn), sum(nu_T6n)/(8*nn)])
U2 = U*U
delta = sum(abs(U2_teor - U2))/6
k = (sum(U2*T)/6)/(sum(T*T)/6)
sluch = ((((sum(U2*U2))/(sum(T*T))) - k**2)/5)
sist = k*(delta/(241.35**2))

print(delta_sist1, delta_sist2, delta_sist3, delta_sist4, delta_sist5, delta_sist6)
print(delta_sluch1, delta_sluch2, delta_sluch3, delta_sluch4, delta_sluch5, delta_sluch6)
print(sigma1, sigma2, sigma3, sigma4, sigma5, sigma6)
print(k, sist, sluch)