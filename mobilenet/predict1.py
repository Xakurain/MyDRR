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
    topk2_label = {}
    pre_value = {}
    topk2_value = {}
    net1.eval()
    with torch.no_grad():
        output_r = net1(img.to(device))
        for label0 in ['rx', 'ry', 'rz']:
            values, indices = output_r[label0].cpu().topk(2, dim=1, largest=True, sorted=True)
            pre_label[label0] = label_dict[label0][str(indices.numpy()[0][0])]
            topk2_label[label0] = label_dict[label0][str(indices.numpy()[0][1])]
            pre_value[label0] = values.numpy()[0][0]
            topk2_value[label0] = values.numpy()[0][1]
    net2.eval()
    with torch.no_grad():
        output_tx = net2(img.to(device))
        values, indices = output_tx['tx'].cpu().topk(2, dim=1, largest=True, sorted=True)
        pre_label['tx'] = label_dict['tx'][str(indices.numpy()[0][0])]
        topk2_label['tx'] = label_dict['tx'][str(indices.numpy()[0][1])]
        pre_value['tx'] = values.numpy()[0][0]
        topk2_value['tx'] = values.numpy()[0][1]
    net3.eval()
    with torch.no_grad():
        output_ty = net3(img.to(device))
        values, indices = output_ty['ty'].cpu().topk(2, dim=1, largest=True, sorted=True)
        pre_label['ty'] = label_dict['ty'][str(indices.numpy()[0][0])]
        topk2_label['ty'] = label_dict['ty'][str(indices.numpy()[0][1])]
        pre_value['ty'] = values.numpy()[0][0]
        topk2_value['ty'] = values.numpy()[0][1]
    net4.eval()
    with torch.no_grad():
        output_tz = net4(img.to(device))
        values, indices = output_tz['tz'].cpu().topk(2, dim=1, largest=True, sorted=True)
        pre_label['tz'] = label_dict['tz'][str(indices.numpy()[0][0])]
        topk2_label['tz'] = label_dict['tz'][str(indices.numpy()[0][1])]
        pre_value['tz'] = values.numpy()[0][0]
        topk2_value['tz'] = values.numpy()[0][1]

    # print('pre_rx: {} pre_ry: {} pre_rz: {} pre_tx: {} pre_ty: {} pre_tz: {}'.format(
    #     pre_label['rx'], pre_label['ry'], pre_label['rz'], pre_label['tx'], pre_label['ty'], pre_label['tz']
    # ))
    return pre_label, topk2_label, pre_value, topk2_value
if __name__ == '__main__':
    img_name = 'F:\\dataset\\imia\\zyt303\\DRRs\\new_DRRs\\img_16458.png'
    pre_label, topk2_label, pre_value, topk2_value = main(img_name)
    print(pre_label)
    x_pre = []
    value_pre = []
    value_topk2 = []
    topk2 = []
    for key in pre_label.keys():
        x_pre.append(int(pre_label[key]))
        topk2.append(int(topk2_label[key]))
        value_pre.append(pre_value[key])
        value_topk2.append(topk2_value[key])
    print(x_pre)
    print(topk2)
    with open('imagelabel_new1.json', 'r') as f:
        label_dict = json.load(f)
    img_num = img_name.split('\\')[-1]
    img_num = img_num.split('.')[0]
    print(label_dict[img_num])