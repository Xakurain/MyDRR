import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from MyDataset import FashionDataset, AttributesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


from model import Myresnet34, get_loss, calculate_metrics


def main(n_epochs, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    attributes = AttributesDataset("F:\\dataset\\imia\\zyt303\\DRRs\\test.csv")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = "F:\\dataset\\imia\\zyt303\\DRRs"
    train_dataset = FashionDataset(os.path.join(data_root, "train.csv"),
                                   attributes, data_transform['train'])

    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=0)
    n_train_samples = len(train_loader)
    print(n_train_samples)

    validate_dataset = FashionDataset(os.path.join(data_root, "val.csv"),
                                   attributes, data_transform['val'])
    val_num = len(validate_dataset)

    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=0)
    n_validate_samples = len(validate_loader)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = Myresnet34(num_rx_classes=17,
                     num_ry_classes=17,
                     num_rz_classes=9,
                     num_tx_classes=41,
                     num_ty_classes=17,
                     num_tz_classes=41).to(device)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "F:\\code\\python\\iMIA\\MyDRR\\resnet\\pth\\resnet34-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 0)
    # net.to(device)

    # 冻结参数
    for name, param in net.named_parameters():
        if "rx" in name:
            param.requires_grad = False
        if "ry" in name:
            param.requires_grad = False
        if "rz" in name:
            param.requires_grad = False
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    best_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    save_root = "F:\\code\\python\\iMIA\\MyDRR\\resnet\\pth"
    print("Starting training ...")
    for epoch in range(n_epochs):
        # train
        total_loss = 0
        net.train()
        batch_num = 0
        for batch in train_loader:
            optimizer.zero_grad()
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            # print(target_labels)
            output = net(img.to(device))
            loss_train, losses_train = get_loss(output, target_labels)
            total_loss += loss_train.item()
            loss_train.backward()
            optimizer.step()

            print("train epoch[{}/{}] batch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                          n_epochs,
                                                          batch_num+1,
                                                          n_train_samples,
                                                          loss_train))
            batch_num += 1
        # validate
        net.eval()
        loss_val = 0
        batch_num
        acc = [0, 0, 0, 0, 0, 0]
        labels = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
        with torch.no_grad():
            for batch in validate_loader:
                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                val_output = net(img.to(device))
                val_train, val_train_losses = get_loss(val_output, target_labels)
                loss_val += val_train.item()
                print("valid epoch[{}/{}] batch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                              n_epochs,
                                                              batch_num + 1,
                                                              n_validate_samples,
                                                              loss_val))
                batch_num += 1

                batch_acc = calculate_metrics(val_output, target_labels)
                acc = [acc[i] + batch_acc[i] for i in range(len(acc))]

                batch_acc_rx, batch_acc_ry, batch_acc_rz, batch_acc_tx, batch_acc_ty, batch_acc_tz = \
                    calculate_metrics(val_output, target_labels)


        val_acc_avg = [acc[i] / n_validate_samples for i in range(len(acc))]
        print('[epoch %d] train_loss: %.3f  val_acc_rx: %.3f val_acc_ry: %.3f val_acc_rz: %.3f\n'
              '                             val_acc_tx: %.3f val_acc_ty: %.3f val_acc_tz: %.3f' %
              epoch + 1, total_loss / n_train_samples, val_acc_avg[0], val_acc_avg[1], val_acc_avg[2],
              val_acc_avg[3], val_acc_avg[4], val_acc_avg[5])

        for i in range(len(acc)):
            if val_acc_avg[i] > best_acc[i]:
                best_acc[i] = val_acc_avg[i]
                torch.save(net.state_dict(), os.path.join(save_root, 'resnet34_' + labels[i] + '.pth'))

    torch.save(net.state_dict(), os.path.join(save_root, 'resnet34_last.pth'))
    print('Finished Training')


if __name__ == '__main__':
    main(50, 32)