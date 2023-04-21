import json
import numpy as np

np.random.seed(17)
inds_rx = np.random.randint(-98, -81, 35000)
np.random.seed(16)
inds_ry = np.random.randint(-8, 9, 35000)
np.random.seed(15)
inds_rz = np.random.randint(-4, 5, 35000)
np.random.seed(14)
inds_tx = np.random.randint(-10, 10, 35000)
inds_tx = inds_tx * 2
np.random.seed(13)
inds_ty = np.random.randint(-4, 5, 35000)
inds_ty = inds_ty * 2
np.random.seed(12)
inds_tz = np.random.randint(-10, 10, 35000)
inds_tz = inds_tz * 2

img_info = {}
for i in range(35000):
    img_name = f'img_{i}'
    img_label = {
        'rx': int(inds_rx[i]),
        'ry': int(inds_ry[i]),
        'rz': int(inds_rz[i]),
        'tx': int(inds_tx[i]),
        'ty': int(inds_ty[i]),
        'tz': int(inds_tz[i])}
    img_info[img_name] = img_label

with open('imagelabel_new1.json','w') as f:
    json.dump(img_info,f,ensure_ascii=False)



# dic_ = {}
# for ind in inds_rx:
#     dic_[ind] = dic_.get(ind, 0) + 1
# print(dic_)









