import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import math
from modules import contour
from model_v3 import mobilenet_v3_large
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # img_true_rgb = cv2.resize(img_true_rgb, (img_pred_rgb.shape[1], img_pred_rgb.shape[0]))
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
    return [img_rgb.shape[1], img_rgb.shape[0]]

def ImagePross(img_back, img_contour, seg=False, contourType='sobel'):
    if seg:
        img_back = contour.otsu_thresholding(img_back)
        img_contour = contour.otsu_thresholding(img_contour)
        img_back = contour.map_binary_to_original(img_back)
        img_contour = contour.map_binary_to_original(img_contour)
    else:
        pass

    segmented_image = torch.from_numpy(img_contour).float().to(device)
    if contourType == 'Sobel':
        img_contour_G = contour.sobel_edge_detection(segmented_image)
        img_contour = img_contour_G.cpu().numpy()
        del img_contour_G, segmented_image
    elif contourType == 'Prewitt':
        img_contour_G = contour.prewitt_edge_detection(segmented_image)
        img_contour = img_contour_G.cpu().numpy()
        del img_contour_G, segmented_image
    elif contourType == 'Roberts Cross':
        img_contour_G = contour.roberts_cross_operator(segmented_image)
        img_contour = img_contour_G.cpu().numpy()
        del img_contour_G, segmented_image
    elif contourType == 'Kirsch':
        img_contour_G = contour.kirsch_operator(segmented_image)
        img_contour = img_contour_G.cpu().numpy()
        del img_contour_G, segmented_image
    elif contourType == 'Canny':
        img_contour_G = contour.canny_edge_detection(segmented_image)
        img_contour = img_contour_G.cpu().numpy()
        del img_contour_G, segmented_image
    else:
        return None
    # 叠加轮廓
    img_bgr = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    # 调整维度一致
    img_bgr = cv2.resize(img_bgr, (img_contour.shape[1], img_contour.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_bgr[img_contour > 50] = [0, 0, 255]
    cv2.imwrite('img_rgb3.jpg', img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def showImage2(parent, img_true, img_pred, mode='differ', seg=False, contourType=None):
    img_rgb = None
    if mode == 'differ':
        img_true = cv2.imread(img_true)
        img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2RGB)
        img_pred = cv2.imread(img_pred)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
        img_rgb = img_pred - img_true
        img_rgb = cv2.normalize(img_rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    elif mode == 'contour1':
        img_true = cv2.imread(img_true)
        img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        img_pred = cv2.imread(img_pred)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
        img_rgb = ImagePross(img_true, img_pred, seg, contourType)
        if img_rgb is None:
            return    
    elif mode == 'contour2':
        img_true = cv2.imread(img_true)
        img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        img_pred = cv2.imread(img_pred)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
        img_rgb = ImagePross(img_pred, img_true, seg, contourType)
        if img_rgb is None:
            return
    else:
        return
    
    qt_img = QImage(img_rgb.data,  # 数据源
                    img_rgb.shape[1],  # 宽度
                    img_rgb.shape[0],  # 高度
                    img_rgb.shape[1] * 3, #行字节数
                    QImage.Format_RGB888)

    pix_img = QPixmap.fromImage(qt_img).scaled(400, 400, aspectMode=Qt.KeepAspectRatio)
    parent.setScaledContents(True)
    parent.setPixmap(pix_img)
    return [img_rgb.shape[1], img_rgb.shape[0]]


def predictinit():
    # print('predictinit')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_r = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21,label_choose=['rx', 'ry', 'rz'])
    net_tx = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21,label_choose=['tx'])
    net_ty = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21,label_choose=['ty'])
    net_tz = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21, label_choose=['tz'])

    model_weight_path1 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_r\\mobilenet_r_last.pth"
    model_weight_path2 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_tx\\mobilenet_tx_tx.pth"
    model_weight_path3 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_ty\\mobilenet_ty_ty.pth"
    model_weight_path4 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_tz\\mobilenet_tz_tz.pth"
    net_r.load_state_dict(torch.load(model_weight_path1, map_location=device))
    net_tx.load_state_dict(torch.load(model_weight_path2, map_location=device))
    net_ty.load_state_dict(torch.load(model_weight_path3, map_location=device))
    net_tz.load_state_dict(torch.load(model_weight_path4, map_location=device))
    net_r.to(device)
    net_tx.to(device)
    net_ty.to(device)
    net_tz.to(device)
    return net_r, net_tx, net_ty, net_tz, device


def predict(net_r, net_tx, net_ty, net_tz, device, img_path):
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

    pre_label = {}
    topk2_label = {}
    pre_value = {}
    topk2_value = {}
    net_r.eval()
    with torch.no_grad():
        output_r = net_r(img.to(device))
        for label0 in ['rx', 'ry', 'rz']:
            values, indices = output_r[label0].cpu().topk(2, dim=1, largest=True, sorted=True)
            pre_label[label0] = label_dict[label0][str(indices.numpy()[0][0])]
            topk2_label[label0] = label_dict[label0][str(indices.numpy()[0][1])]
            pre_value[label0] = values.numpy()[0][0]
            topk2_value[label0] = values.numpy()[0][1]
    net_tx.eval()
    with torch.no_grad():
        output_tx = net_tx(img.to(device))
        values, indices = output_tx['tx'].cpu().topk(2, dim=1, largest=True, sorted=True)
        pre_label['tx'] = label_dict['tx'][str(indices.numpy()[0][0])]
        topk2_label['tx'] = label_dict['tx'][str(indices.numpy()[0][1])]
        pre_value['tx'] = values.numpy()[0][0]
        topk2_value['tx'] = values.numpy()[0][1]

    net_ty.eval()
    with torch.no_grad():
        output_ty = net_ty(img.to(device))
        values, indices = output_ty['ty'].cpu().topk(2, dim=1, largest=True, sorted=True)
        pre_label['ty'] = label_dict['ty'][str(indices.numpy()[0][0])]
        topk2_label['ty'] = label_dict['ty'][str(indices.numpy()[0][1])]
        pre_value['ty'] = values.numpy()[0][0]
        topk2_value['ty'] = values.numpy()[0][1]

    net_tz.eval()
    with torch.no_grad():
        output_tz = net_tz(img.to(device))
        values, indices = output_tz['tz'].cpu().topk(2, dim=1, largest=True, sorted=True)
        pre_label['tz'] = label_dict['tz'][str(indices.numpy()[0][0])]
        topk2_label['tz'] = label_dict['tz'][str(indices.numpy()[0][1])]
        pre_value['tz'] = values.numpy()[0][0]
        topk2_value['tz'] = values.numpy()[0][1]

    return pre_label, topk2_label, pre_value, topk2_value

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

