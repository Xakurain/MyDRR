import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from new_MyDataset import FashionDataset, AttributesDataset
from model_v3 import mobilenet_v3_large


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes_path', default='F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\split\\test.csv')
    parser.add_argument('--data_root', default='F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\split')
    parser.add_argument('--save_root', default='F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num_rx_classes', type=int, default=17)
    parser.add_argument('--num_ry_classes', type=int, default=17)
    parser.add_argument('--num_rz_classes', type=int, default=9)
    parser.add_argument('--num_tx_classes', type=int, default=21)
    parser.add_argument('--num_ty_classes', type=int, default=9)
    parser.add_argument('--num_tz_classes', type=int, default=21)
    parser.add_argument('--net_name', default='resnet34')
    return parser.parse_args()

def main(args, label_choose):
    n_epochs = args.epochs
    batch_size = args.batch_size
    attributes_path = args.attributes_path
    attributes = AttributesDataset(attributes_path, label_choose)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_root  # 分组文件存在的地方
    train_dataset = FashionDataset(os.path.join(data_root, "test.csv"),
                                   attributes, label_choose, data_transform['train'])

    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=0)
    n_train_samples = len(train_loader)
    print(n_train_samples)

    for epoch in range(n_epochs):
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            img = data['img']
            target_labels = data['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            print(target_labels)
if __name__ == '__main__':
    args = parse_args()
    label_choose = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
    main(args, label_choose)