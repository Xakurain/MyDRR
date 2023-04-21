import numpy
import json
# f=open('F:\\code\\python\\iMIA\\MyDRR\\test\\1.txt', encoding='utf8')
# txt=[]
# for line in f:
#     txt.append(line.strip())
# print(txt)

# for i in range(len(txt)):
#     txt[i] = txt[i][0:-4]
# print(txt)
rx =[]
ry =[]
rz =[]
tx =[]
ty =[]
tz =[]
for rx1 in range(-98, -81, 1):
    rx.append(rx1)
    # print(rx1)
    
for ry1 in range(-8, 9, 1):
    ry.append(ry1)
    # print(ry1)

for rz1 in range(-4, 5, 1):
    rz.append(rz1)
    # print(rz1)

for tx1 in range(-20, 20, 1):
    tx.append(tx1)
    # print(tx1)

for ty1 in range(-8, 9, 1):
    ty.append(ty1)
    # print(ty1)   

for tz1 in range(-20, 20, 1):
    tz.append(tz1)
    # print(tz1)  
print(rx, len(rx))
print(ry, len(ry))
print(rz, len(rz))
print(tx, len(tx))
print(ty, len(ty))
print(tz, len(tz))

