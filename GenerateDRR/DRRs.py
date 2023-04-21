
import os
from pathlib import Path
import sys
import json
import time
sys.path.append('../wrapped_modules/')
from DRRGenerate import pyDRRGenerate 


dcmfilepath = "F:\\dataset\\imia\\zyt303\\ScalarVolume_16"
p = pyDRRGenerate(dcmfilepath, False)
p.ReadDCM()
# dic = {"DRRs":{"moving_image": [], "fixed_image": []}}

rx = -90.
ry = 0.
rz = 0.
tx = 0.
ty = 0.
tz = 0.
sid = 400
sx = 3
sy = 3
dx = 512
dy = 512
threshold = 0

output_root = "F:\\dataset\\imia\\zyt303\\"
img_num = 0
count = 31000
with open('F:\\code\\python\\iMIA\\MyDRR\\imagelabel_new1.json','r')  as f:
    dic_img = json.load(f)
    start = time.time()
    print('start')
    for img_num in range(31000, 35000):
        img_num_s = f"img_{img_num}"
        dic_img1 = dic_img[img_num_s]
        rx = dic_img1['rx']
        ry = dic_img1['ry']
        rz = dic_img1['rz']
        tx = dic_img1['tx']
        ty = dic_img1['ty']
        tz = dic_img1['tz']
        save_dcm_name = f'{output_root}\\DRRs\\new_DRRs\\img_{img_num}.png'
        p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)
        count += 1
        if count % 500 == 0:
            end = time.time()
            print("已生成{}张图片，总用时{:.2f}s".format(count, end-start))


        

# dic = {"DRRs":{"moving_image": [], "fixed_image": []}}
# for rx1 in range(-98, -81, 4):
#     rx = rx1
#     for ry1 in range(-8, 9, 4):
#         ry = ry1
#         for rz1 in range(-4, 5, 4):
#             rz = rz1
#             for tx1 in range(-20, 20, 4):
#                 tx = tx1
#                 for ty1 in range(-8, 9, 4):
#                     ty = ty1
#                     for tz1 in range(-20, 20, 4):
#                         tz = tz1
#                         img_name = f"img_{img_num}"
#                         img_dic = {img_name:{"rx":rx,
#                                             "ry":ry, 
#                                             "rz": rz, 
#                                             "tx": tx,
#                                             "ty": ty,
#                                             "tz": tz}}
#                         dic["DRRs"]["moving_image"].append(img_dic)
#                         dic["DRRs"]["fixed_image"].append(img_dic)
#                         save_dcm_name = f'{output_root}png\\DRRs\\img_{img_num}.png'
#                         p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)
#                         img_num += 1
#                         if img_num == 15000:
#                             break


# with open('tttt.json','w') as f:
#     json.dump(dic,f,ensure_ascii=False)


# for rx1 in range(-98, -81):
#     rx = rx1
#     save_dcm_name = f'{output_root}png\\rx\\rx_{rx}.png'
#     p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)

# for ry1 in range(-8, 9):
#     ry = ry1
#     save_dcm_name = f'{output_root}png\\ry\\ry_{ry}.png'
#     p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)

# for rz1 in range(-8, 9):
#     rz = rz1
#     save_dcm_name = f'{output_root}png\\rz\\rz_{rz}.png'
#     p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)
# for tz1 in range(-20, 21):
#     tz = tz1
#     save_dcm_name = f'{output_root}png\\tz\\tz_{tz}.png'
#     p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)

# for tx1 in range(-20, 21):
#     tx = tx1
#     save_dcm_name = f'{output_root}png\\tx\\tx_{tx}.png'
#     p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)

# for ty1 in range(-20, 21):
#     ty = ty1
#     save_dcm_name = f'{output_root}png\\ty\\ty_{ty}.png'
#     p.Drr1(save_dcm_name, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)

