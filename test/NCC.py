import cv2
import numpy as np
import math
#归一化互相关系数
def NCC(y_true_path, y_pred_path):
    y_true = cv2.imread(y_true_path, 0)
    y_pred = cv2.imread(y_pred_path, 0)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    NCC = cov / (std_true * std_pred)
    return NCC


if __name__ == '__main__':
    # y_true_path = 'F:\\dataset\\imia\\zyt303\drr\\DRRs\\DRRs\\img_0.png'
    # y_pred_path = 'F:\\dataset\\imia\\zyt303\drr\\DRRs\\DRRs\\img_0.png'
    y_true_path = 'F:\\dataset\\imia\\zyt303\drr\\rx\\rx_-53.png'
    y_pred_path = 'F:\\dataset\\imia\\zyt303\drr\\rx\\rx_-53.png'
    print(NCC(y_true_path, y_pred_path))