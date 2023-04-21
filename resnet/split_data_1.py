# split_data.py
import argparse
import csv
import os
import json
import numpy as np

def save_csv(data, path, fieldnames = ['image_path', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='/content/drive/MyDrive/MyDRR/imagelabel_new1.json')
    parser.add_argument('--output_folder', type=str, default='/content/drive/MyDrive/MyDRR/resnet/split')
    parser.add_argument('--image_folder', type=str, default='/content/DRRs')
    parser.add_argument('--img_num', type=int, default=35000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    image_folder = args.image_folder

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
    inds = np.random.choice(args.img_num, args.img_num, replace=False)
    num_1 = int(0.95 * args.img_num)
    num_2 = int(0.80 * num_1)
    print(num_2)
    print(num_1)

    save_csv(all_data[inds][:num_2], os.path.join(output_folder, 'train.csv'))
    save_csv(all_data[inds][num_2:num_1], os.path.join(output_folder, 'val.csv'))
    save_csv(all_data[inds][num_1:args.img_num], os.path.join(output_folder, 'test.csv'))

