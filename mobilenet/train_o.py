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
from loss import get_loss, calculate_metrics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes_path', default='F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\split\\test.csv')
    parser.add_argument('--data_root', default='F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\split')
    parser.add_argument('--save_root', default='F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num_rx_classes', type=int, default=17)
    parser.add_argument('--num_ry_classes', type=int, default=17)
    parser.add_argument('--num_rz_classes', type=int, default=9)
    parser.add_argument('--num_tx_classes', type=int, default=21)
    parser.add_argument('--num_ty_classes', type=int, default=9)
    parser.add_argument('--num_tz_classes', type=int, default=21)
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
    train_dataset = FashionDataset(os.path.join(data_root, "train.csv"),
                                   attributes, label_choose, data_transform['train'])

    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=0)
    n_train_samples = len(train_loader)
    print(n_train_samples)

    validate_dataset = FashionDataset(os.path.join(data_root, "val.csv"),
                                      attributes, label_choose, data_transform['val'])
    val_num = len(validate_dataset)

    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=0)
    n_validate_samples = len(validate_loader)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = mobilenet_v3_large(num_rx_classes=args.num_rx_classes,
                             num_ry_classes=args.num_ry_classes,
                             num_rz_classes=args.num_rz_classes,
                             num_tx_classes=args.num_tx_classes,
                             num_ty_classes=args.num_ty_classes,
                             num_tz_classes=args.num_tz_classes,
                             label_choose=label_choose)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    # model_weight_path = "/content/drive/MyDrive/MyDRR/mobilenet/mobilenet.pth"
    # assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    # pre_weights = torch.load(model_weight_path, map_location='cpu')
    #
    # # delete classifier weights
    # pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    # for param in net.features.parameters():
    #     param.requires_grad = False

    net.to(device)

    # define loss function
    # loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    best_acc = [0.0 for i in range(len(label_choose))]
    save_root = args.save_root
    print("Starting training ...")
    for epoch in range(n_epochs):
        # train
        net.train()
        total_loss = 0
        batch_num = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            img = data['img']
            target_labels = data['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}

            output = net(img.to(device))
            optimizer.zero_grad()
            loss_train, losses_train = get_loss(output, target_labels, label_choose)
            total_loss += loss_train.item()
            loss_train.backward()
            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] batch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     n_epochs,
                                                                     step + 1,
                                                                     n_train_samples,
                                                                     loss_train)

        # validate
        net.eval()
        acc = [0.0 for i in range(len(label_choose))]  # accumulate accurate number / epoch

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                img = val_data['img']
                target_labels = val_data['labels']
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                val_output = net(img.to(device))
                val_train, val_train_losses = get_loss(val_output, target_labels, label_choose)

                val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                            n_epochs,
                                                                            val_train)
                batch_acc = calculate_metrics(val_output, target_labels, label_choose)
                acc = [acc[i] + batch_acc[i] for i in range(len(acc))]

        val_acc_avg = [acc[i] / n_validate_samples for i in range(len(acc))]

        print_out = f'[epoch {epoch + 1}] train_loss: {total_loss / n_train_samples}'
        for i in range(len(label_choose)):
            print_out += f' {label_choose[i]}_acc: {val_acc_avg[i]}'
            if val_acc_avg[i] >= best_acc[i]:
                best_acc[i] = val_acc_avg[i]
                torch.save(net.state_dict(), os.path.join(save_root, 'mobilenet_t_' + label_choose[i] + '.pth'))
        print(print_out)
        torch.save(net.state_dict(), os.path.join(save_root, 'mobilenet_t_last.pth'))
    print('Finished Training')


if __name__ == '__main__':
    label_choose = ['rx', 'ry', 'rz']
    args = parse_args()
    main(args, label_choose)
