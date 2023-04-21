import numpy as np
import cv2
from PIL import Image

y_true_path = 'F:\\dataset\\imia\\zyt303\drr\\DRRs\\DRRs\\img_0.png'
y_pred_path = 'F:\\dataset\\imia\\zyt303\drr\\DRRs\\DRRs\\img_0.png'
# y_true_path = 'F:\\dataset\\imia\\zyt303\drr\\rx\\rx_-53.png'
# y_pred_path = 'F:\\dataset\\imia\\zyt303\drr\\rx\\rx_-54.png'
# y_true = cv2.imread(y_true_path, cv2.IMREAD_GRAYSCALE)
# y_pred = cv2.imread(y_pred_path, cv2.IMREAD_GRAYSCALE)
#
# y_true = y_true/255
# y_pred = y_pred/255
#
# u = y_true*y_pred
# dice = 2*np.sum(u) / (np.sum(y_true)+np.sum(y_pred))
# print(dice)

s2 = cv2.imread(y_true_path, 0)# 模板
row, col = s2.shape[0], s2.shape[1]
s1 = cv2.imread(y_pred_path, 0)
s = []
for r in range(row):
    for c in range(col):
        if s1[r][c] == s2[r][c]: # 计算图像像素交集
            s.append(s1[r][c])
#                 print(s1[r][c])
m1 = np.linalg.norm(s)
m2 = np.linalg.norm(s1.flatten()) + np.linalg.norm(s2.flatten())
d=(2*m1/m2)
print(d)
