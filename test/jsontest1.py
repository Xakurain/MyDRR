import json
import os
import csv
input_folder = "F:\\code\\python\\iMIA\\MyDRR\\imagelable.json"
image_folder = "F:\\dataset\\imia\\zyt303\\drr\\DRRs\\DRRs"
# with open(input_folder, 'r') as f:
#     dic_img = json.load(f)
#     all_data = []
#
#     for img_num, value in dic_img.items():
#         rx = value['rx']
#         ry = value['ry']
#         rz = value['rz']
#         tx = value['tx']
#         ty = value['ty']
#         tz = value['tz']
#
#         img_name = os.path.join(image_folder, img_num + '.png')
#
#         if os.path.exists(img_name):
#             all_data.append([img_name, rx, ry, rz, tx, ty, tz])

with open('F:\\dataset\\imia\\zyt303\\drr\\DRRs\\val.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['image_path'])




