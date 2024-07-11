import numpy as np
import cv2
from PIL import Image

y_true_path = 'F:\\dataset\\imia\\zyt303\\DRRs\\new_DRRs\\img_15604.png'
y_pred_path = 'F:\\code\\python\\iMIA\\MyDRR\\test\\y_pred3.png'
def dice(y_true_path, y_pred_path):
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
    return d
