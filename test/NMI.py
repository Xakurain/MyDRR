import numpy as np
import cv2
import math

# y_true_path = 'F:\\dataset\\imia\\zyt303\drr\\rx\\rx_-53.png'
# y_pred_path = 'F:\\dataset\\imia\\zyt303\drr\\rx\\rx_-54.png'
y_true_path = 'F:\\dataset\\imia\\zyt303\drr\\DRRs\\DRRs\\img_0.png'
y_pred_path = 'F:\\dataset\\imia\\zyt303\drr\\DRRs\\DRRs\\img_0.png'

y_true = cv2.imread(y_true_path, 0)
y_pred = cv2.imread(y_pred_path, 0)

high, width = y_true.shape
tmp = np.zeros((256, 256))
H_XY = 0
H_true = 0
H_pred = 0
for i in range(high):
    for j in range(width):
        val1 = y_true[i][j]
        val2 = y_pred[i][j]
        tmp[val1][val2] = float(tmp[val1][val2]) + 1
tmp = tmp / (high * width)
tmp_true = np.sum(tmp, axis=1)
tmp_pred = np.sum(tmp, axis=0)
for i in range(256):
    for j in range(256):
        if tmp[i][j] != 0:
            H_XY = H_XY - tmp[i][j] * math.log2(tmp[i][j])
    if tmp_true[i] != 0:
        H_true = H_true - tmp_true[i] * math.log2(tmp_true[i])
    if tmp_pred[i] != 0:
        H_pred = H_pred - tmp_pred[i] * math.log2(tmp_pred[i])
MI = H_true + H_pred - H_XY
H_max = max(H_true, H_pred)
NMI = MI / H_max
print(NMI)




