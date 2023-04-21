from MyDataset import AttributesDataset
import json

attributes = AttributesDataset("F:\\dataset\\imia\\zyt303\\drr\\DRRs\\test.csv")
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

