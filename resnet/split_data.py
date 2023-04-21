# split_data.py
import argparse
import csv
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def save_csv(data, path, fieldnames = ['image_path', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))

if __name__ == '__main__':
    input_folder = "F:\\code\\python\\iMIA\\MyDRR\\imagelabel_new.json"
    output_folder = "F:\\dataset\\imia\\zyt303\\DRRs"
    image_folder = "F:\\dataset\\imia\\zyt303\\DRRs\\DRRs"

    all_data = []
    with open(input_folder, 'r') as f:
        dic_img = json.load(f)

        for img_num, value in dic_img.items():
            rx = value['rx']
            ry = value['ry']
            rz = value['rz']
            tx = value['tx']
            ty = value['ty']
            tz = value['tz']

            img_name = os.path.join(image_folder, img_num + '.png')

            if os.path.exists(img_name):
                all_data.append([img_name, rx, ry, rz, tx, ty, tz])

    np.random.seed(17)

    all_data = np.asarray(all_data)
    inds = np.random.choice(30000, 30000, replace=False)

    save_csv(all_data[inds][:22800], os.path.join(output_folder, 'train.csv'))
    save_csv(all_data[inds][22800:28500], os.path.join(output_folder, 'val.csv'))
    save_csv(all_data[inds][28500:30000], os.path.join(output_folder, 'test.csv'))


