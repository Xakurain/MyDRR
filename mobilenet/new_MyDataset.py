
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class AttributesDataset():
    def __init__(self, annotation_path, label_choose):
        rx_labels = []
        ry_labels = []
        rz_labels = []
        tx_labels = []
        ty_labels = []
        tz_labels = []
        self.label_choose = label_choose

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rx_labels.append(row['rx'])
                ry_labels.append(row['ry'])
                rz_labels.append(row['rz'])
                tx_labels.append(row['tx'])
                ty_labels.append(row['ty'])
                tz_labels.append(row['tz'])

        self.rx_labels = np.unique(rx_labels)
        self.ry_labels = np.unique(ry_labels)
        self.rz_labels = np.unique(rz_labels)
        self.tx_labels = np.unique(tx_labels)
        self.ty_labels = np.unique(ty_labels)
        self.tz_labels = np.unique(tz_labels)

        self.num_rx = len(self.rx_labels)
        self.num_ry = len(self.ry_labels)
        self.num_rz = len(self.rz_labels)
        self.num_tx = len(self.tx_labels)
        self.num_ty = len(self.ty_labels)
        self.num_tz = len(self.tz_labels)

        self.rx_id_to_name = dict(zip(range(len(self.rx_labels)), self.rx_labels))
        self.rx_name_to_id = dict(zip(self.rx_labels, range(len(self.rx_labels))))

        self.ry_id_to_name = dict(zip(range(len(self.ry_labels)), self.ry_labels))
        self.ry_name_to_id = dict(zip(self.ry_labels, range(len(self.ry_labels))))

        self.rz_id_to_name = dict(zip(range(len(self.rz_labels)), self.rz_labels))
        self.rz_name_to_id = dict(zip(self.rz_labels, range(len(self.rz_labels))))

        self.tx_id_to_name = dict(zip(range(len(self.tx_labels)), self.tx_labels))
        self.tx_name_to_id = dict(zip(self.tx_labels, range(len(self.tx_labels))))

        self.ty_id_to_name = dict(zip(range(len(self.ty_labels)), self.ty_labels))
        self.ty_name_to_id = dict(zip(self.ty_labels, range(len(self.ty_labels))))

        self.tz_id_to_name = dict(zip(range(len(self.tz_labels)), self.tz_labels))
        self.tz_name_to_id = dict(zip(self.tz_labels, range(len(self.tz_labels))))

class FashionDataset(Dataset):
    def __init__(self, annotation_path, attributes, label_choose, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes
        # 初始化数组以存储真实标签和图像路径
        self.data = []

        if 'rx' in label_choose:
            self.rx_labels = []
        if 'ry' in label_choose:
            self.ry_labels = []
        if 'rz' in label_choose:
            self.rz_labels = []
        if 'tx' in label_choose:
            self.tx_labels = []
        if 'ty' in label_choose:
            self.ty_labels = []
        if 'tz' in label_choose:
            self.tz_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                if 'rx' in label_choose:
                    self.rx_labels.append(self.attr.rx_name_to_id[row['rx']])
                if 'ry' in label_choose:
                    self.ry_labels.append(self.attr.ry_name_to_id[row['ry']])
                if 'rz' in label_choose:
                    self.rz_labels.append(self.attr.rz_name_to_id[row['rz']])
                if 'tx' in label_choose:
                    self.tx_labels.append(self.attr.tx_name_to_id[row['tx']])
                if 'ty' in label_choose:
                    self.ty_labels.append(self.attr.ty_name_to_id[row['ty']])
                if 'tz' in label_choose:
                    self.tz_labels.append(self.attr.tz_name_to_id[row['tz']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        dict_data = {
            'img': img,
            'labels': {}
        }
        if 'rx' in self.attr.label_choose:
            dict_data['labels']['rx'] = self.rx_labels[idx]
        if 'ry' in self.attr.label_choose:
            dict_data['labels']['ry'] = self.ry_labels[idx]
        if 'rz' in self.attr.label_choose:
            dict_data['labels']['rz'] = self.rz_labels[idx]
        if 'tx' in self.attr.label_choose:
            dict_data['labels']['tx'] = self.tx_labels[idx]
        if 'ty' in self.attr.label_choose:
            dict_data['labels']['ty'] = self.ty_labels[idx]
        if 'tz' in self.attr.label_choose:
            dict_data['labels']['tz'] = self.tz_labels[idx]

        return dict_data