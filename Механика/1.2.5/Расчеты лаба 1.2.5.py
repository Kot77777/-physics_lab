import numpy as np

Omega1 = 0.01*np.array([21.26, 21.17, 20, 21.4, 20.8])
Omega1_sr = 0.01*20.93
M1 = 0.01*40.898

Omega2 = 0.01*np.array([16.34, 16.69, 16.88, 16.67, 16.86])
Omega2_sr = 0.01*16.69
M2 = 0.01*32.428

Omega3 = 0.01*np.array([13.42, 13.46, 13.41, 13.41, 13.46])
Omega3_sr = 0.01*13.43
M3 = 0.01*26.015

Omega4 = 0.01*np.array([10.9, 10.88, 10.9, 10.86, 10.9])
Omega4_sr = 0.01*10.89
M4 = 0.01*20.933

Omega5 = 0.01*np.array([8.82, 8.84, 8.86, 8.85, 8.85])
Omega5_sr = 0.01*8.84
M5 = 0.01*17.182

om_sl1 = (0.05*(sum((np.array([Omega1[i] - Omega1_sr for i in range(5)]))**2)))**0.5
om_sl2 = (0.05*(sum((np.array([Omega2[i] - Omega2_sr for i in range(5)]))**2)))**0.5
om_sl3 = (0.05*(sum((np.array([Omega3[i] - Omega3_sr for i in range(5)]))**2)))**0.5
om_sl4 = (0.05*(sum((np.array([Omega4[i] - Omega4_sr for i in range(5)]))**2)))**0.5
om_sl5 = (0.05*(sum((np.array([Omega5[i] - Omega5_sr for i in range(5)]))**2)))**0.5

print(om_sl1/0.2093, om_sl2/0.1669, om_sl3/0.1343, om_sl4/0.1089, om_sl5/0.0884)

M = np.array([M1, M2, M3, M4, M5])
Omega = np.array([Omega1_sr, Omega2_sr, Omega3_sr, Omega4_sr, Omega5_sr])
M_sr = sum(M)/5
M_kv_sr = sum(M**2)/5
M_Omega = sum(M*Omega)/5
Omega_kv_sr = sum(Omega**2)/5
k = M_Omega / M_kv_sr
print(k)
print((0.25*((Omega_kv_sr)/(M_kv_sr) - k**2))**0.5)