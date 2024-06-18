import torch
import torch.nn.functional as F
from utils import read_data, load_nii, minmax
import numpy as np
import nibabel as nib


# 加载测试数据
t2_path = "/data/jiazhen/code/3DClassification-framework/dataset/lesions_classification_dataset/Lesions/CHEN_JING_YUAN/T2W.nii.gz"

t2_image = torch.tensor(load_nii(t2_path).astype(np.float32))       # (18, 432, 432)
# t2_image = t2_image.unsqueeze(0)
t2_size = t2_image.permute(1, 2, 0).shape

map_path = '/data/jiazhen/code/3DClassification-framework/attention_maps/features.transition3.conv/attention_map_0_0_0.nii.gz'
map_image = torch.tensor(load_nii(map_path).astype(np.float32))     # (28,2,14)
map_image = map_image.permute(1, 0, 2)    # (14,2,28)

map_image = map_image.unsqueeze(0).unsqueeze(0)

map_image_new = F.interpolate(map_image, size=(224,224,32), mode='trilinear', align_corners=True)

map_image_new = map_image_new.squeeze(0).squeeze(0)
map_image_new = map_image_new.numpy()
nii = nib.Nifti1Image(map_image_new, np.eye(4))

nib.save(nii, './attention_maps/output.nii.gz')






