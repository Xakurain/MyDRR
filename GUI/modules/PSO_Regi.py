from datetime import datetime

import numpy as np
import math
import sys
import cv2
import os
sys.path.append('../wrapped_modules/')
from DRRGenerate import pyDRRGenerate



def Dice(y_true, y_pred):
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

#粒子群算法
#更新每个粒子的速度向量
def update_velocity(v_i, p_i, g, x_i, w, c1, c2, v_max):
    """
        根据速度更新公式更新每个粒子的速度
        :param V: 粒子当前的速度矩阵
        :param X: 粒子当前的位置矩阵
        :param p: 每个粒子历史最优位置
        :param g: 种群历史最优位置
        """
    r1 = np.random.random(1)
    r2 = np.random.random(1)
    v_i = w * v_i + c1 * r1 * (p_i - x_i) + c2 * r2 * (g - x_i)
    # 速度限制
    v_i[v_i > v_max] = v_max
    v_i[v_i < -v_max] = -v_max
    return v_i
#更新每个粒子的位置向量
def update_position(x, v, x_max, x_min):
    x = x + v
    # 位置限制
    for i in range(len(x)):
        if x[i] > x_max:
            x[i] = x_max
        if x[i] < x_min:
            x[i] = x_min
    return x

#计算适应度函数, 目标函数值
def fitness(x, ndim, x_pred, ct, y_true):
    x = x[0]
    x_pred1 = x_pred.copy()
    x_pred1[ndim] = x
    curtime = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    y_pred_name = f'F:\\code\\python\\iMIA\\MyDRR\\GUI\\img\\y_pred_{curtime}.png'
    ct.Drr1(y_pred_name, x_pred1[0], x_pred1[1], x_pred1[2], x_pred1[3], x_pred1[4], x_pred1[5], 400, 3, 3, 512, 512, 0)
    while not os.path.exists(y_pred_name):
        pass
    y_pred = cv2.imread(y_pred_name, 0)
    os.remove(y_pred_name)
    return Dice(y_true, y_pred)

def PSO(ct, x_top1, x_top2, value_top1, value_top2, y_true_path):
    sub_value_index = np.argsort(np.subtract(value_top1, value_top2))
    ndim = sub_value_index[0]
    x_topk2 = np.array([x_top1, x_top2])
    x_pred = x_topk2[0]
    x_pred = x_pred.astype(np.float32)
    x_pred_new = x_pred.copy()
    x_max = x_topk2.max(axis=0)
    x_min = x_topk2.min(axis=0)

    particle_num = 10 # 粒子数量
    dim = 1 # 粒子维度
    v_max = 0.2 # 最大速度
    iter_num = 3 # 迭代次数
    w = 0.3 # 惯性权重
    c1 = 2 # 个体学习因子
    c2 = 2 # 社会学习因子
    y_true = cv2.imread(y_true_path, 0)
    g_value = fitness([x_pred[ndim]],ndim, x_pred, ct, y_true)

    epoch = 0
    while(g_value < 0.8):
        # print(f'epoch: {epoch}')
        ndim = sub_value_index[epoch]
        if ndim >= 3:
            v_max = 0.4
        # 初始化粒子群
        x = np.random.uniform(x_min[ndim], x_max[ndim], (particle_num, dim))
        v = np.random.uniform(-v_max, v_max, (particle_num, dim))
        x[0] = x_pred_new[ndim]
        p = x.copy()
        fitness_p = []
        for i in range(particle_num):
            fitness_p.append(fitness(p[i], ndim, x_pred_new, ct, y_true))
        fitness_p = np.array(fitness_p)
        g = p[np.argmax(fitness_p)]
        g_value = np.max(fitness_p)

        # 迭代寻优
        for i in range(iter_num):
            for j in range(particle_num):
                v[j] = update_velocity(v[j], p[j], g, x[j], w, c1, c2, v_max)
                x[j] = update_position(x[j], v[j], x_max[ndim], x_min[ndim])
                x_value = fitness(x[j], ndim, x_pred_new, ct, y_true)
                p_value = fitness(p[j], ndim, x_pred_new, ct, y_true)
                if x_value > p_value:
                    p[j] = x[j]
                    p_value = x_value
                if p_value > g_value:
                    g = p[j]
                    g_value = p_value
        x_pred_new[ndim] = g[0]
        if epoch >= 1:
            # 判断是否保留上一轮的最优值
            fitness_last = fitness(g, ndim, x_pred, ct, y_true)
            if fitness_last > g_value or math.fabs(g_value-fitness_last) < 0.07:
                x_pred_new[sub_value_index[epoch-1]] = x_pred[sub_value_index[epoch-1]]
                g_value = fitness_last
            else:
                x_pred = x_pred_new.copy()
        epoch += 1
        if epoch > 5:
            break
    x_pred_new = np.around(x_pred_new, decimals=4)
    return x_pred_new
