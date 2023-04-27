import os
import csv
import json
from PIL import Image
import torch
from torchvision import transforms
from model_v3 import mobilenet_v3_large


def main(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    with open('classlabel.json', 'r') as f:
        label_dict = json.load(f)

    # load image
    img_path = img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    net1 = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21,label_choose=['rx', 'ry', 'rz'])
    net2 = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21,label_choose=['tx'])
    net3 = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21,label_choose=['ty'])
    net4 = mobilenet_v3_large(num_rx_classes=17,num_ry_classes=17, num_rz_classes=9,num_tx_classes=21,num_ty_classes=9,
                              num_tz_classes=21, label_choose=['tz'])

    model_weight_path1 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_r\\mobilenet_r_last.pth"
    model_weight_path2 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_tx\\mobilenet_tx_tx.pth"
    model_weight_path3 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_ty\\mobilenet_ty_ty.pth"
    model_weight_path4 = "F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_tz\\mobilenet_tz_tz.pth"
    net1.load_state_dict(torch.load(model_weight_path1, map_location=device))
    net2.load_state_dict(torch.load(model_weight_path2, map_location=device))
    net3.load_state_dict(torch.load(model_weight_path3, map_location=device))
    net4.load_state_dict(torch.load(model_weight_path4, map_location=device))
    net1.to(device)
    net2.to(device)
    net3.to(device)
    net4.to(device)

    pre_label = {}
    net1.eval()
    with torch.no_grad():
        output_r = net1(img.to(device))
        for label0 in ['rx', 'ry', 'rz']:
            _, predicted_label0 = output_r[label0].cpu().max(1)
            pre_label[label0] = label_dict[label0][str(predicted_label0.numpy()[0])]
    net2.eval()
    with torch.no_grad():
        output_tx = net2(img.to(device))
        _, predicted_label1 = output_tx['tx'].cpu().max(1)
        pre_label['tx'] = label_dict['tx'][str(predicted_label1.numpy()[0])]
    net3.eval()
    with torch.no_grad():
        output_ty = net3(img.to(device))
        _, predicted_label2 = output_ty['ty'].cpu().max(1)
        pre_label['ty'] = label_dict['ty'][str(predicted_label2.numpy()[0])]
    net4.eval()
    with torch.no_grad():
        output_tz = net4(img.to(device))
        _, predicted_label3 = output_tz['tz'].cpu().max(1)
        pre_label['tz'] = label_dict['tz'][str(predicted_label3.numpy()[0])]
    # print('pre_rx: {} pre_ry: {} pre_rz: {} pre_tx: {} pre_ty: {} pre_tz: {}'.format(
    #     pre_label['rx'], pre_label['ry'], pre_label['rz'], pre_label['tx'], pre_label['ty'], pre_label['tz']
    # ))
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
    with open('F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\split\\test.csv') as f:
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
        pre_img = main(data_path[i])
        tru_img = {'rx': rx_labels[i], 'ry': ry_labels[i], 'rz': rz_labels[i], 'tx': tx_labels[i], 'ty': ty_labels[i], 'tz': tz_labels[i]}
        for label0 in pre_img:
            if pre_img[label0] == tru_img[label0]:
                pre_acc_num[label0] += 1
            else:
                err_img.append(data_path[i])
        if(i % 100 == 0):
            print('已处理{}张'.format(i))
    pre_acc = {}
    for label0, num in pre_acc_num.items():
        pre_acc[label0] = num / len(data_path)
    print('acc_rx: %.3f acc_ry: %.3f acc_rz: %.3f acc_tx: %.3f acc_ty: %.3f acc_tz: %.3f' %
          (pre_acc['rx'], pre_acc['ry'], pre_acc['rz'], pre_acc['tx'], pre_acc['ty'], pre_acc['tz']))
    print('err_img: {}'.format(err_img))
    print('err_num: {}'.format(len(err_img)))

    img_path = 'F:\\dataset\\imia\\zyt303\\DRRs\\new_DRRs\\img_1.png'
    main(img_path)
