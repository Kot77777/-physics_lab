sp2 = [476, 400, 288, 192, 108, 116, 208, 308, 388, 484]
sp1 = [147.2 ,124.1 ,89.0 ,59.6 ,33.3 ,35.1 ,64.4, 95.5, 120.2, 150]
total1 = total2 = total3 = total4 = total5 = 0
for i in range(10):
    total1+=sp1[i]
    total2+=sp2[i]
    total3+=(sp1[i])**2
    total4+=(sp2[i])**2
    total5+=(sp1[i]*sp2[i])
print(total1/10)
print(total2/10)
print(total3/10)
print(total4/10)
print(total5/10)














