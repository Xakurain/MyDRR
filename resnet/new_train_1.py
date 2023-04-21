import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from new_MyDataset import FashionDataset, AttributesDataset
from torch.utils.data import DataLoader
import math
from tqdm import tqdm


from new_model import Myresnet34, Myresnet50, get_loss, calculate_metrics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes_path', default='/content/drive/MyDrive/MyDRR/resnet/split/test.csv')
    parser.add_argument('--data_root', default='/content/drive/MyDrive/MyDRR/resnet/split')
    parser.add_argument('--save_root', default='/content/drive/MyDrive/MyDRR/resnet/pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_root
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

    if args.net_name == 'resnet34':
        net = Myresnet34(num_rx_classes=args.num_rx_classes,
                         num_ry_classes=args.num_ry_classes,
                         num_rz_classes=args.num_rz_classes,
                         num_tx_classes=args.num_tx_classes,
                         num_ty_classes=args.num_ty_classes,
                         num_tz_classes=args.num_tz_classes,
                         label_choose=label_choose)
    elif args.net_name == 'resnet50':
        net = Myresnet50(num_rx_classes=args.num_rx_classes,
                         num_ry_classes=args.num_ry_classes,
                         num_rz_classes=args.num_rz_classes,
                         num_tx_classes=args.num_tx_classes,
                         num_ty_classes=args.num_ty_classes,
                         num_tz_classes=args.num_tz_classes,
                         label_choose=label_choose)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "/content/drive/MyDrive/MyDRR/resnet/resnet34_new_t_pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 0)
    net.to(device)

    # # 冻结参数
    # for name, param in net.named_parameters():
    #     if "rx" in name:
    #         param.requires_grad = False
    #     if "ry" in name:
    #         param.requires_grad = False
    #     if "rz" in name:
    #         param.requires_grad = False

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = [0.0 for i in range(len(label_choose))]
    save_root = args.save_root
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
            loss_train, losses_train = get_loss(output, target_labels, label_choose)
            total_loss += loss_train.item()
            loss_train.backward()
            optimizer.step()

            print("train epoch[{}/{}] batch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                          n_epochs,
                                                          batch_num+1,
                                                          n_train_samples,
                                                          loss_train))
            batch_num += 1
        scheduler.step()
        # validate
        net.eval()
        batch_num  = 0
        acc = [0.0 for i in range(len(label_choose))]
        labels = label_choose
        with torch.no_grad():
            for batch in validate_loader:
                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                val_output = net(img.to(device))
                val_train, val_train_losses = get_loss(val_output, target_labels, label_choose)
                print("valid epoch[{}/{}] batch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                              n_epochs,
                                                              batch_num + 1,
                                                              n_validate_samples,
                                                              val_train))
                batch_num += 1

                batch_acc = calculate_metrics(val_output, target_labels, label_choose)
                acc = [acc[i] + batch_acc[i] for i in range(len(acc))]

        val_acc_avg = [acc[i] / n_validate_samples for i in range(len(acc))]
        print_out = f'[epoch {epoch+1}] train_loss: {total_loss / n_train_samples}'
        for i in range(len(label_choose)):
            print_out += f' {labels[i]}_acc: {val_acc_avg[i]}'
            if val_acc_avg[i] >= best_acc[i]:
                best_acc[i] = val_acc_avg[i]
                torch.save(net.state_dict(), os.path.join(save_root, 'resnet34_new_t_' + labels[i] + '.pth'))
        print(print_out)
    torch.save(net.state_dict(), os.path.join(save_root, 'resnet34_new_t_last.pth'))
    print('Finished Training')


if __name__ == '__main__':
    label_choose = ['tx', 'ty', 'tz']
    args = parse_args()
    main(args, label_choose)

