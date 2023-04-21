import os
import csv
import json
import torch
from PIL import Image
from torchvision import transforms
from model import Myresnet34

def predict(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with open('classlabel.json', 'r') as f:
        label_dict = json.load(f)

    label_lst = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
    # load image
    img_path = img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    net = Myresnet34(num_rx_classes=5,
                     num_ry_classes=5,
                     num_rz_classes=3,
                     num_tx_classes=10,
                     num_ty_classes=5,
                     num_tz_classes=10)

    model_weight_path = "F:\\code\\python\\iMIA\\MyDRR\\resnet\\resnet34_last.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net.to(device)

    net.eval()
    with torch.no_grad():
        output = net(img.to(device))
        pre_label = {}
        for label0 in label_lst:
            _, predicted_label0 = output[label0].cpu().max(1)
            pre_label0 = label_dict[label0][str(predicted_label0.tolist()[0])]
            pre_label[label0] = pre_label0

        print('pre_rx: {} pre_ry: {} pre_rz: {} pre_tx: {} pre_ty: {} pre_tz: {}'.format(
            pre_label['rx'], pre_label['ry'], pre_label['rz'], pre_label['tx'], pre_label['ty'], pre_label['tz']
        ))
        return pre_label

if __name__ == '__main__':
    data_path = []
    rx_labels = []
    ry_labels = []
    rz_labels = []
    tx_labels = []
    ty_labels = []
    tz_labels = []
    err_img = []
    pre_acc_num = {'rx': 0, 'ry': 0, 'rz': 0, 'tx': 0, 'ty': 0, 'tz': 0}
    with open('F:\\dataset\\imia\\zyt303\\drr\\DRRs\\test.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_path.append(row['image_path'])
            rx_labels.append(row['rx'])
            ry_labels.append(row['ry'])
            rz_labels.append(row['rz'])
            tx_labels.append(row['tx'])
            ty_labels.append(row['ty'])
            tz_labels.append(row['tz'])


    for i in range(len(data_path)):
        pre_img = predict(data_path[i])
        tru_img = {'rx': rx_labels[i], 'ry': ry_labels[i], 'rz': rz_labels[i], 'tx': tx_labels[i], 'ty': ty_labels[i], 'tz': tz_labels[i]}
        for label0 in pre_img:
            if pre_img[label0] == tru_img[label0]:
                pre_acc_num[label0] += 1
            else:
                err_img.append(i+1)
    pre_acc = {}
    for label0, num in pre_acc_num.items():
        pre_acc[label0] = num/len(data_path)

    print('acc_rx: %.3f acc_ry: %.3f acc_rz: %.3f acc_tx: %.3f acc_ty: %.3f acc_tz: %.3f' %
        (pre_acc['rx'], pre_acc['ry'], pre_acc['rz'], pre_acc['tx'], pre_acc['ty'], pre_acc['tz']))
    print(err_img)




