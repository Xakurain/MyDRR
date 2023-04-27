from new_MyDataset import AttributesDataset
import json

attributes = AttributesDataset("F:\\code\\python\\iMIA\\MyDRR\\mobilenet\\result_tx\\test.csv", ['rx', 'ry', 'rz', 'tx', 'ty', 'tz'])
label_dict = {
    'rx': attributes.rx_id_to_name,
    'ry': attributes.ry_id_to_name,
    'rz': attributes.rz_id_to_name,
    'tx': attributes.tx_id_to_name,
    'ty': attributes.ty_id_to_name,
    'tz': attributes.tz_id_to_name
}
with open('classlabel.json','w') as f:
    json.dump(label_dict,f,ensure_ascii=False)

