"""
The dataset of the Lesion dataset
Add 2 sequences to the two branches of the model
"""

import os
import numpy as np
import torch
# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
# from get_dataset_folder import get_brats_folder
# from utils import pad_or_crop_image, minmax, load_nii, pad_image_and_label, listdir
from utils import read_data, load_nii, minmax
from monai.transforms import Compose, RandRotate90, RandSpatialCrop, EnsureChannelFirst, RandFlip, RandGaussianNoise, Resize, NormalizeIntensity, RandGaussianSmoothd, RandAdjustContrastd



class leisions(Dataset):
    def __init__(self, patients_dir, mode, transform=None):
        # super(BraTS, self).__init__()
        self.patients_dir = patients_dir
        self.mode = mode
        self.transform = transform

        self.data = []
        with open(patients_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                label = items[0]
                folder_name = items[1].split('/')[-1]
                folder_path = items[1]
                self.data.append((label, folder_path, folder_name))

    def __getitem__(self, idx):
        label, folder_path, folder_name = self.data[idx]
        t1_path = os.path.join(folder_path, 'T1W.nii.gz')
        t2_path = os.path.join(folder_path, 'T2W.nii.gz')

        t1_image = torch.tensor(load_nii(t1_path).astype(np.float32))
        t1_image = t1_image.unsqueeze(0)

        t2_image = torch.tensor(load_nii(t2_path).astype(np.float32))
        t2_image = t2_image.unsqueeze(0)

        #Removing black area from the edge of the MRI
        nonzero_index = torch.nonzero(torch.sum(t1_image, axis=0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:,0], nonzero_index[:,1], nonzero_index[:,2]
        zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        t1_image = t1_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()
        t1_image = t1_image.permute(0, 3, 2, 1)

        nonzero_index = torch.nonzero(torch.sum(t2_image, axis=0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
        zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        t2_image = t2_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()
        t2_image = t2_image.permute(0, 3, 2, 1)

        for i in range(t1_image.shape[0]):
            t1_image[i] = minmax(t1_image[i])

        for i in range(t2_image.shape[0]):
            t2_image[i] = minmax(t2_image[i])

        if label == "Lesions":
            label = 1
        else:
            label = 0

        t1_image = self.transform(t1_image)
        t2_image = self.transform(t2_image)

        return t1_image, t2_image, label

    def __len__(self):
        return len(self.data)


def get_lesions_datasets(dataset_folder, mode, transform):
    dataset_folder = os.path.join(dataset_folder, mode + '.txt')
    assert os.path.exists(dataset_folder), "Dataset Folder Does Not Exist1"
    output = leisions(patients_dir=dataset_folder, mode=mode, transform=transform)

    return output


if __name__ == "__main__":
    train_dataset = get_lesions_datasets("./dataset/brats2021", "train")
    train_val_dataset = get_lesions_datasets("./dataset/brats2021", "val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, drop_last=True)
    train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, data in enumerate(train_loader):
        label = data["label"].cuda()
        images = data["image"].cuda()
        # images, label = data_aug(images, label)
        # pred = model(images)
        # pred = F.interpolate(pred, size=label.size()[2:], mode='nearest')
        # train_loss = criterion(pred, label)
        # train_loss_meter.update(train_loss.item())
        # train_loss.backward()
        # optimizer.step()

