import itk
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

import random

import ProjectorsModule as pm
import ReadWriteImageModule as rw

model_filepath="F:\\dataset\\imia\\zyt303_1\\303 Calf_0.nii.gz"
focal_lenght = 500
Projector_info = {'Name': 'SiddonGpu',
                  'threadsPerBlock_x': 16,
                  'threadsPerBlock_y': 16,
                  'threadsPerBlock_z': 1,
                  'focal_lenght': focal_lenght,
                  'DRRspacing_x': 0.2756, # 0.5, 1
                  'DRRspacing_y': 0.2756,
                  'DRR_ppx': 3.6180,
                  'DRR_ppy': 3.6180,
                  'DRRsize_x': 1024,
                  'DRRsize_y': 1024,
                  }
projector = pm.projector_factory(Projector_info, model_filepath)
drr1 = projector.compute([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0, 0, 0])
plt.subplot(111)
plt.imshow(drr1, cmap='gray')
plt.show()


