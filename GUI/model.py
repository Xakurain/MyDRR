import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import warnings

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out



class MyResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_rx_classes,
                 num_ry_classes,
                 num_rz_classes,
                 num_tx_classes,
                 num_ty_classes,
                 num_tz_classes,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(MyResNet, self).__init__()

        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # self.fc = nn.Linear(512 * block.expansion, 0)
            self.rx = nn.Linear(512 * block.expansion, num_rx_classes)
            self.ry = nn.Linear(512 * block.expansion, num_ry_classes)
            self.rz = nn.Linear(512 * block.expansion, num_rz_classes)
            self.tx = nn.Linear(512 * block.expansion, num_tx_classes)
            self.ty = nn.Linear(512 * block.expansion, num_ty_classes)
            self.tz = nn.Linear(512 * block.expansion, num_tz_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # x = self.fc(x)

        return {
            'rx': self.rx(x),
            'ry': self.ry(x),
            'rz': self.rz(x),
            'tx': self.tx(x),
            'ty': self.ty(x),
            'tz': self.tz(x)
        }


def Myresnet34(num_rx_classes,
               num_ry_classes,
               num_rz_classes,
               num_tx_classes,
               num_ty_classes,
               num_tz_classes,
               include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return MyResNet(BasicBlock, [3, 4, 6, 3],
                  num_rx_classes,
                  num_ry_classes,
                  num_rz_classes,
                  num_tx_classes,
                  num_ty_classes,
                  num_tz_classes,
                  include_top=include_top)

def Myresnet50(num_rx_classes,
               num_ry_classes,
               num_rz_classes,
               num_tx_classes,
               num_ty_classes,
               num_tz_classes,
               include_top=True):

    return MyResNet(Bottleneck, [3, 4, 6, 3],
                  num_rx_classes,
                  num_ry_classes,
                  num_rz_classes,
                  num_tx_classes,
                  num_ty_classes,
                  num_tz_classes,
                  include_top=include_top)

def get_loss(net_output, ground_truth):
    rx_loss = F.cross_entropy(net_output['rx'], ground_truth['rx'])
    ry_loss = F.cross_entropy(net_output['ry'], ground_truth['ry'])
    rz_loss = F.cross_entropy(net_output['rz'], ground_truth['rz'])
    tx_loss = F.cross_entropy(net_output['tx'], ground_truth['tx'])
    ty_loss = F.cross_entropy(net_output['ty'], ground_truth['ty'])
    tz_loss = F.cross_entropy(net_output['tz'], ground_truth['tz'])

    loss = tx_loss + ty_loss + tz_loss + rx_loss + ry_loss + rz_loss

    return loss, {'rx': rx_loss,
                  'ry': ry_loss,
                  'rz': rz_loss,
                  'tx': tx_loss,
                  'ty': ty_loss,
                  'tz': tz_loss}

def calculate_metrics(output, target):
    _, predicted_rx = output['rx'].cpu().max(1)
    gt_rx = target['rx'].cpu()

    _, predicted_ry = output['ry'].cpu().max(1)
    gt_ry = target['ry'].cpu()

    _, predicted_rz = output['rz'].cpu().max(1)
    gt_rz = target['rz'].cpu()

    _, predicted_tx = output['tx'].cpu().max(1)
    gt_tx = target['tx'].cpu()

    _, predicted_ty = output['ty'].cpu().max(1)
    gt_ty = target['ty'].cpu()

    _, predicted_tz = output['tz'].cpu().max(1)
    gt_tz = target['tz'].cpu()

    with warnings.catch_warnings():  # sklearn 在处理混淆矩阵中的零行时可能会产生警告
        warnings.simplefilter("ignore")
        acc_rx = balanced_accuracy_score(y_true=gt_rx.numpy(), y_pred=predicted_rx.numpy())
        acc_ry = balanced_accuracy_score(y_true=gt_ry.numpy(), y_pred=predicted_ry.numpy())
        acc_rz = balanced_accuracy_score(y_true=gt_rz.numpy(), y_pred=predicted_rz.numpy())
        acc_tx = balanced_accuracy_score(y_true=gt_tx.numpy(), y_pred=predicted_tx.numpy())
        acc_ty = balanced_accuracy_score(y_true=gt_ty.numpy(), y_pred=predicted_ty.numpy())
        acc_tz = balanced_accuracy_score(y_true=gt_tz.numpy(), y_pred=predicted_tz.numpy())

    return [acc_rx, acc_ry, acc_rz, acc_tx, acc_ty, acc_tz]