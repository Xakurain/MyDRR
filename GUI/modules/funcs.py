import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import os
import csv
import json
import torch
from PIL import Image
from torchvision import transforms
import sys
import numpy as np
import math
from model import Myresnet34


def showImage(parent, img, mode='path'):
    if mode == 'path':
        img_bgr = cv2.imread(img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif mode == 'difference':
        img_true, img_pred = img
        img_true_bgr = cv2.imread(img_true)
        img_true_rgb = cv2.cvtColor(img_true_bgr, cv2.COLOR_BGR2RGB)
        img_pred_bgr = cv2.imread(img_pred)
        img_pred_rgb = cv2.cvtColor(img_pred_bgr, cv2.COLOR_BGR2RGB)
        # 调整参考图像尺寸与预测图像尺寸一致
        img_true_rgb = cv2.resize(img_true_rgb, (img_pred_rgb.shape[1], img_pred_rgb.shape[0]))
        img_rgb = img_pred_rgb - img_true_rgb
        img_rgb = cv2.normalize(img_rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    qt_img = QImage(img_rgb.data,  # 数据源
                    img_rgb.shape[1],  # 宽度
                    img_rgb.shape[0],  # 高度
                    # img_rgb.shape[1] * 3, #行字节数
                    QImage.Format_RGB888)

    pix_img = QPixmap.fromImage(qt_img).scaled(400, 400, aspectMode=Qt.KeepAspectRatio)
    parent.setScaledContents(True)
    parent.setPixmap(pix_img)

def predictinit():
    print('predictinit')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Myresnet34(num_rx_classes=5,
                     num_ry_classes=5,
                     num_rz_classes=3,
                     num_tx_classes=10,
                     num_ty_classes=5,
                     num_tz_classes=10)

    model_weight_path = "F:\\code\\python\\iMIA\\MyDRR\\resnet\\resnet34_last.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net.to(device)
    return net, device


def predict(net, device, img_path):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with open('classlabel.json', 'r') as f:
        label_dict = json.load(f)

    label_lst = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
    # load image
    img_path = img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    net.eval()
    with torch.no_grad():
        output = net(img.to(device))
        pre_label = {}
        for label0 in label_lst:
            _, predicted_label0 = output[label0].cpu().max(1)
            pre_label0 = label_dict[label0][str(predicted_label0.tolist()[0])]
            pre_label[label0] = pre_label0

        print('pre_rx: {} pre_ry: {} pre_rz: {} pre_tx: {} pre_ty: {} pre_tz: {}'.format(
            pre_label['rx'], pre_label['ry'], pre_label['rz'], pre_label['tx'], pre_label['ty'], pre_label['tz']
        ))
        return pre_label

#Dice系数
def dice(y_true, y_pred):
    row, col = y_true.shape[0], y_true.shape[1]
    s = []
    for r in range(row):
        for c in range(col):
            if y_pred[r][c] == y_true[r][c]:  # 计算图像像素交集
                s.append(y_pred[r][c])
    #                 print(s1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(y_pred.flatten()) + np.linalg.norm(y_true.flatten())
    d = (2 * m1 / m2)
    return d

#归一化互相关
def NCC(y_true, y_pred):

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    NCC = cov / (std_true * std_pred)
    return NCC

#标准化互信息
def NMI(y_true, y_pred):

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
    return NMI

def Analysis(y_true_path, y_pred_path):
    y_true = cv2.imread(y_true_path, 0)
    y_pred = cv2.imread(y_pred_path, 0)
    # 调整参考图像尺寸与配准图像尺寸一致
    y_true = cv2.resize(y_true, (y_pred.shape[1], y_pred.shape[0]), interpolation=cv2.INTER_CUBIC)
    d = dice(y_true, y_pred)
    ncc = NCC(y_true, y_pred)
    nmi = NMI(y_true, y_pred)
    return d, ncc, nmi



