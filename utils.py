import os
import sys
import pickle
import random

import SimpleITK
import numpy as np
import torch

from networks import new_model, new_model_1, new_model_2, new_model_3, ResNet3D, ViT3D, DenseNet3D, C3DNet, SqueezeNet3D, MobileNet3D, MobileNetV2_3D, ResNeXt3D, ShuffleNet3D, ShuffleNetV2_3D

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_score



def read_data(root):
    # assert os.path.exist(root), "dataset root: {} does not exist.".format(root)

    patient_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    patient_class.sort()

    patients_path = []
    labels_path = []

    for cla in patient_class:
        cla_path = os.path.join(root, cla)
        for patient_id in os.listdir(cla_path):
            patient_path = os.path.join(cla_path, patient_id)

            patients_path.append(patient_path)
            labels_path.append(cla)
    return patients_path, labels_path


def load_nii(path):
    nii_file = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(path)))
    return nii_file

def normalize(image):
    min_ = torch.min(image)
    max_ = torch.max(image)
    scale_ = max_ - min_
    image = (image - min_) / scale_
    return image

def minmax(image, low_perc=1, high_perc=99):
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = torch.clip(image, low, high)
    image = normalize(image)
    return image


def get_model(args):
    if args.model_name == 'new_model':
        model = new_model.generate_model(model_depth=args.model_depth, n_input_channels=args.n_input_channels, n_classes=args.n_classes)
    elif args.model_name == 'ResNet3D':
        model = ResNet3D.generate_model(model_depth=args.model_depth, n_input_channels=args.n_input_channels, n_classes=args.n_classes)
    elif args.model_name == 'ResNeXt3D':
        model = ResNeXt3D.generate_model(model_depth=args.model_depth, sample_size=args.height, sample_duration=args.depth, n_classes=args.n_classes, in_channels=args.n_input_channels)
    elif args.model_name == 'ViT3D':
        model = ViT3D.ViT3D(image_size=(448, 448, 32), patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, channels=2, dropout=0.1, emb_dropout=0.1)
    elif args.model_name == 'DenseNet3D':
        model = DenseNet3D.generate_model(model_depth=args.model_depth, n_input_channels=args.n_input_channels, num_classes=args.n_classes)
    elif args.model_name == 'C3DNet':
        model = C3DNet.generate_model(num_classes=args.n_classes, sample_size=args.height, sample_duration=args.depth, in_channels=args.n_input_channels)
    elif args.model_name == 'SqueezeNet3D':
        model = SqueezeNet3D.generate_model(version=1.1, num_classes=args.n_classes, sample_size=args.height, sample_duration=args.depth, in_channels=args.n_input_channels)
    elif args.model_name == 'MobileNet3D':
        model = MobileNet3D.generate_model(num_classes=args.n_classes, sample_size=args.height, width_mult=1, in_channels=args.n_input_channels)
    elif args.model_name == 'MobileNetV2_3D':
        model = MobileNetV2_3D.generate_model(num_classes=args.n_classes, sample_size=args.height, width_mult=1, in_channels=args.n_input_channels)
    elif args.model_name == 'ShuffleNet3D':
        model = ShuffleNet3D.generate_model(groups=3, num_classes=args.n_classes, width_mult=1, in_channels=args.n_input_channels)
    elif args.model_name == 'ShuffleNetV2_3D':
        model = ShuffleNetV2_3D.generate_model(num_classes=args.n_classes, sample_size=args.height, width_mult=1., in_channels=args.n_input_channels)
    elif args.model_name == 'new_model_1':
        model = new_model_1.generate_model(model_depth=args.model_depth, n_input_channels=args.n_input_channels, n_classes=args.n_classes)
    elif args.model_name == 'new_model_2':
        model = new_model_2.generate_model(model_depth=args.model_depth, n_input_channels=args.n_input_channels, n_classes=args.n_classes)
    elif args.model_name == 'new_model_3':
        model = new_model_3.generate_model(model_depth=args.model_depth, n_input_channels=args.n_input_channels, n_classes=args.n_classes)
    else:
        print("There is no such model!")
    return model






class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def pre_cal(total_cnf_matrix):
    num_classes = total_cnf_matrix.shape[0]

    # 初始化存储每个类别指标的列表
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    f1_score_list = []

    for i in range(num_classes):
        # 计算每个类别的准确率
        class_accuracy = total_cnf_matrix[i, i] / (np.sum(total_cnf_matrix[i, :]) + 1e-6)
        accuracy_list.append(class_accuracy)

        # 计算每个类别的敏感性（召回率）
        true_positive = total_cnf_matrix[i, i]
        false_negative = np.sum(total_cnf_matrix[i, :]) - true_positive
        class_sensitivity = true_positive / ((true_positive + false_negative) + 1e-6)
        sensitivity_list.append(class_sensitivity)

        # 计算每个类别的特异性
        true_negative = np.sum(np.delete(np.delete(total_cnf_matrix, i, axis=0), i, axis=1))
        false_positive = np.sum(total_cnf_matrix[:, i]) - true_positive
        class_specificity = true_negative / ((true_negative + false_positive) + 1e-6)
        specificity_list.append(class_specificity)

        # 计算每个类别的精确率
        false_positive = np.sum(total_cnf_matrix[:, i]) - true_positive
        class_precision = true_positive / ((true_positive + false_positive) + 1e-6)
        precision_list.append(class_precision)

        # 计算每个类别的F1分数
        class_f1_score = 2 * (class_precision * class_sensitivity) / ((class_precision + class_sensitivity) + 1e-6)
        f1_score_list.append(class_f1_score)

    # 计算总准确率
    total_accuracy = round(np.trace(total_cnf_matrix) / (np.sum(total_cnf_matrix) + 1e-6), 6)

    return total_accuracy, accuracy_list, sensitivity_list, specificity_list, precision_list, f1_score_list



