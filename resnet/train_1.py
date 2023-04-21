import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from MyDataset import FashionDataset, AttributesDataset
from torch.utils.data import DataLoader
import math
from tqdm import tqdm


from model import Myresnet34, Myresnet50, get_loss, calculate_metrics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes_path', default='/content/drive/MyDrive/MyDRR/resnet/split/test.csv')
    parser.add_argument('--data_root', default='/content/drive/MyDrive/MyDRR/resnet/split')
    parser.add_argument('--save_root', default='/content/drive/MyDrive/MyDRR/resnet/pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    return parser.parse_args()

def main(args):

    n_epochs = args.epochs
    batch_size = args.batch_size
    attributes_path = args.attributes_path
    attributes = AttributesDataset(attributes_path)

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
    
    net = Myresnet50(num_rx_classes=17,
                     num_ry_classes=17,
                     num_rz_classes=9,
                     num_tx_classes=41,
                     num_ty_classes=17,
                     num_tz_classes=41)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "/content/drive/MyDrive/MyDRR/resnet/resnet50_new_pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 0)
    net.to(device)

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
    optimizer = optim.Adam(params, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
        scheduler.step()
        # validate
        net.eval()
        batch_num  = 0
        acc = [0, 0, 0, 0, 0, 0]
        labels = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
        with torch.no_grad():
            for batch in validate_loader:
                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                val_output = net(img.to(device))
                val_train, val_train_losses = get_loss(val_output, target_labels)
                print("valid epoch[{}/{}] batch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                              n_epochs,
                                                              batch_num + 1,
                                                              n_validate_samples,
                                                              val_train))
                batch_num += 1

                batch_acc = calculate_metrics(val_output, target_labels)
                acc = [acc[i] + batch_acc[i] for i in range(len(acc))]

        val_acc_avg = [acc[i] / n_validate_samples for i in range(len(acc))]
        print('[epoch %d] train_loss: %.3f  val_acc_rx: %.3f val_acc_ry: %.3f val_acc_rz: %.3f val_acc_tx: %.3f val_acc_ty: %.3f val_acc_tz: %.3f' %
              (epoch + 1, total_loss / n_train_samples, val_acc_avg[0], val_acc_avg[1], val_acc_avg[2],
              val_acc_avg[3], val_acc_avg[4], val_acc_avg[5]))
        for i in range(len(acc)):
            if val_acc_avg[i] >= best_acc[i]:
                best_acc[i] = val_acc_avg[i]
                torch.save(net.state_dict(), os.path.join(save_root, 'resnet50_new_' + labels[i] + '.pth'))
    torch.save(net.state_dict(), os.path.join(save_root, 'resnet50_new_last.pth'))
    print('Finished Training')


if __name__ == '__main__':
    args = parse_args()
    main(args)

