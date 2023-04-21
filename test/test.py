import itk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import random
import argparse
# sys.path.append('../wrapped_modules/')
# from DRRGenerate import pyDRRGenerate
#


# print('ass\n'
#       'sss')
# lst0 = [1, 1, 1, 1, 1, 1, 1]
# lst = [3, 3, 3, 3, 3, 3, 3]
# lst = [lst[i]/3 for i in range(len(lst))]
# lst1 = [lst[i]+lst0[i] for i in range(len(lst))]
# print(lst1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes_path', default='F:\\dataset\\imia\\zyt303\\drr\\DRRs\\test.csv')
    parser.add_argument('--data_root', default='F:\\dataset\\imia\\zyt303\\drr\\DRRs')
    parser.add_argument('--save_root', default='F:\\code\\python\\iMIA\\MyDRR\\resnet\\pth')
    return parser.parse_args()

if __name__ == '__main__':

    # args = parse_args()
    # print(args.attributes_path)
    # print(args.data_root)
    # print(args.save_root)

    # #Drr生成测试
    # dcmfilepath = "F:\\dataset\\imia\\zyt303\\ScalarVolume_16"
    # save_root = "F:\\dataset\\imia\\zyt303\\"
    # p = pyDRRGenerate(dcmfilepath, False)
    # p.ReadDCM()

    # rx = -90.
    # ry = 0.
    # rz = 0.
    # tx = 0.
    # ty = 0.
    # tz = 0.
    # sid = 400
    # sx = 3
    # sy = 3
    # dx = 512
    # dy = 512
    # threshold = 0
    # for tx in range(0, 21):
    #     save_path = save_root + "test\\drr_tx_" + str(tx) + ".png"
    #     p.Drr1(save_path, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)
    # np.random.seed(14)
    # inds_tx = np.random.randint(-10, 11, 100)
    # inds_tx = inds_tx*2
    # dic_ = {}
    # for ind in inds_tx:
    #     dic_[ind] = dic_.get(ind, 0) + 1
    # print(dic_, len(dic_))
    # acc = [0.0 for i in range(3)]
    # print(acc)
    path = 'F:\\dataset\\imia\\zyt303\drr\\DRRs\\DRRs\\img_0.png'
    img_pred_bgr = cv2.imread(path,0)
    # img_pred_rgb = cv2.cvtColor(img_pred_bgr, cv2.COLOR_BGR2RGB)
    print(img_pred_bgr.shape)



