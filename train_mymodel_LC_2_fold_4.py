"""
Import two sequences into two branches separately.
new_model
2023-11-23:Revised the evaluation indicators to address the issue of inaccurate calculation of sensitivity, specificity, and other indicators.
2023-12-08:Add the AUC value and ROC curve figure.
2023-12-09:Swap the position of data elemments. (1, 18, 442, 512) --> (1, 512, 442, 18)
"""

"""
CosineAnnealingLR lr=0.01
"""

import sys
import pandas as pd
import math
import torch
import torch.nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset.brats import get_brats_datasets
from dataset.leisions import get_lesions_datasets
import argparse
import os
from utils import pre_cal, AverageMeter, get_model
from collections import OrderedDict
from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise, Resize, NormalizeIntensity, ToTensor
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main(args):
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    if args.name is None:
        args.name = '%s_%s_%s_train_fold_4_001_20240306' % (args.dataset, args.model_name, args.scheduler)
    os.makedirs('results/%s' % args.name, exist_ok=True)

    # Data loading code
    train_transform = Compose([RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                                Resize(spatial_size=(320, 320, 18)),
                                RandFlip(spatial_axis=(0, 1, 2)),
                                RandGaussianNoise(prob=0.15, mean=0, std=0.33),
                                ToTensor(),
                                NormalizeIntensity(),
                                # RandGaussianSmoothd(prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                                # RandAdjustContrastd(prob=0.15, gamma=(0.7, 1.3)),
                                ])
    val_transform = Compose([Resize(spatial_size=(320, 320, 18)),
                             RandGaussianNoise(prob=0.15, mean=0, std=0.33),
                             ToTensor(),
                             NormalizeIntensity(),
                             # RandGaussianSmoothd(prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                             # RandAdjustContrastd(prob=0.15, gamma=(0.7, 1.3)),
                            ])

    # Data loading code
    train_dataset = get_lesions_datasets(args.dataset_folder, "train_4", transform=train_transform)
    val_dataset = get_lesions_datasets(args.dataset_folder, "val_4", transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             )


    # model = generate_model(model_depth=args.model_depth, n_input_channels=args.n_input_channels, n_classes=args.n_classes)
    model = get_model(args)
    model = model.to(device)

    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                              nesterov=args.nesterov, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    else:
        raise NotImplementedError


    # Define the learning rate strategy.
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs/200, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=1, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    elif args.scheduler == 'LambdaLR':
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        raise NotImplementedError

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('train_loss', []),
        ('train_total_acc', []),
        ('train_acc', []),
        ('train_sensitivity', []),
        ('train_specificity', []),
        ('train_precision', []),
        ('train_f1', []),
        ('val_loss', []),
        ('val_total_acc', []),
        ('val_acc', []),
        ('val_sensitivity', []),
        ('val_specificity', []),
        ('val_precision', []),
        ('val_f1', []),
        ('val_auc', []),
    ])

    best_acc = 0

    # TensorBoard visualization
    os.makedirs('./runs/%s' % args.name, exist_ok=True)
    writer = SummaryWriter(log_dir='./runs/%s' % args.name)

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        train_log = train(model=model,
                          optimizer=optimizer,
                          data_loader=train_loader,
                          device=device,
                          )

        val_log = val(model=model,
                      data_loader=val_loader,
                      device=device,
                      epoch=epoch + 1,
                      )

        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_log['val_loss'])
        elif args.scheduler == 'MultiStepLR':
            scheduler.step()
        elif args.scheduler == 'ConstantLR':
            scheduler.step()
        elif args.scheduler == 'LambdaLR':
            scheduler.step()
        else:
            pass

        log['epoch'].append(epoch + 1)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['train_loss'].append(train_log['train_loss'])
        log['train_total_acc'].append(train_log['train_total_acc'])
        log['train_acc'].append(train_log['train_acc'])
        log['train_sensitivity'].append(train_log['train_sensitivity'])
        log['train_specificity'].append(train_log['train_specificity'])
        log['train_precision'].append(train_log['train_precision'])
        log['train_f1'].append(train_log['train_f1'])

        log['val_loss'].append(val_log['val_loss'])
        log['val_total_acc'].append(val_log['val_total_acc'])
        log['val_acc'].append(val_log['val_acc'])
        log['val_sensitivity'].append(val_log['val_sensitivity'])
        log['val_specificity'].append(val_log['val_specificity'])
        log['val_precision'].append(val_log['val_precision'])
        log['val_f1'].append(val_log['val_f1'])
        log['val_auc'].append(val_log['val_auc'])

        os.makedirs('./results/%s' % args.name, exist_ok=True)
        pd.DataFrame(log).to_csv('./results/%s/log.csv' % args.name, index=False)

        os.makedirs('./weights/%s' % args.name, exist_ok=True)
        if val_log['val_total_acc'] > best_acc:
            if os.path.exists('./weights/{}'.format(args.name)):
                files = [f for f in os.listdir('./weights/{}'.format(args.name)) if f.endswith('.pth')]
                for file in files:
                    os.remove(os.path.join('./weights/{}'.format(args.name), file))
            best_acc = val_log['val_total_acc']
            torch.save(model.state_dict(), './weights/{}/epoch_{}.pth'.format(args.name, epoch))

        # TensorBoard visualization
        writer.add_scalar(tag="lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar(tag="train_loss", scalar_value=train_log['train_loss'], global_step=epoch)
        writer.add_scalar(tag="train_total_acc", scalar_value=train_log['train_total_acc'], global_step=epoch)
        writer.add_scalar(tag="train_sensitivity", scalar_value=train_log['train_sensitivity'][1], global_step=epoch)
        writer.add_scalar(tag="train_specificity", scalar_value=train_log['train_specificity'][1], global_step=epoch)
        writer.add_scalar(tag="train_precision", scalar_value=train_log['train_precision'][1], global_step=epoch)
        writer.add_scalar(tag="train_f1", scalar_value=train_log['train_f1'][1], global_step=epoch)

        writer.add_scalar(tag="val_loss", scalar_value=val_log['val_loss'], global_step=epoch)
        writer.add_scalar(tag="val_total_acc", scalar_value=val_log['val_total_acc'], global_step=epoch)
        writer.add_scalar(tag="val_sensitivity", scalar_value=val_log['val_sensitivity'][1], global_step=epoch)
        writer.add_scalar(tag="val_specificity", scalar_value=val_log['val_specificity'][1], global_step=epoch)
        writer.add_scalar(tag="val_precision", scalar_value=val_log['val_precision'][1], global_step=epoch)
        writer.add_scalar(tag="val_f1", scalar_value=val_log['val_f1'][1], global_step=epoch)
        writer.add_scalar(tag="val_auc", scalar_value=val_log['val_auc'], global_step=epoch)

    writer.close()


def train(model, optimizer, data_loader, device):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    total_cnf_matrix = np.zeros((2, 2))
    iter_num = 0

    all_labels = []
    all_preds = []
    train_loss = []

    pbar = tqdm(total=len(data_loader))
    for step, data in enumerate(data_loader):
        t1_images, t2_images, labels = data
        t1_images = t1_images.to(device).float()
        t2_images = t2_images.to(device).float()
        labels = labels.to(device).long()
        iter_num += 1

        pred = model(t1_images, t2_images).to(device)
        pred_classes = torch.max(pred, dim=1)[1]

        all_labels = np.concatenate([all_labels, labels.cpu().numpy()])
        all_preds = np.concatenate([all_preds, pred_classes.cpu().numpy()])

        total_cnf_matrix += confusion_matrix(labels.cpu().numpy(), pred_classes.cpu().numpy(), labels=[0, 1])
        total_acc, acc, sensitivity, specificity, precision, f1 = pre_cal(total_cnf_matrix)

        loss = loss_function(pred, labels)
        loss.backward()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        train_loss.append(loss.item())
        train_loss_avg = np.sum(train_loss) / iter_num

        postfix = OrderedDict([
            ('train_loss', train_loss_avg),
            ('train_total_acc', total_acc),
            ('train_acc', acc),
            ('train_sensitivity', sensitivity),
            ('train_specificity', specificity),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('train_loss', train_loss_avg),
                        ('train_total_acc', total_acc),
                        ('train_acc', acc),
                        ('train_sensitivity', sensitivity),
                        ('train_specificity', specificity),
                        ('train_precision', precision),
                        ('train_f1', f1)
                        ])



def val(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    total_cnf_matrix = np.zeros((2, 2))
    iter_num = 0

    all_labels = []
    all_preds_probs = []
    val_loss = []

    pbar = tqdm(total=len(data_loader))
    for step, data in enumerate(data_loader):
        t1_images, t2_images, labels = data
        t1_images = t1_images.to(device).float()
        t2_images = t2_images.to(device).float()
        labels = labels.to(device).long()
        iter_num += 1

        pred_probs = torch.softmax(model(t1_images, t2_images).to(device), dim=1)
        pred_classes = torch.max(pred_probs, dim=1)[1]

        all_labels = np.concatenate([all_labels, labels.cpu().numpy()])
        all_preds_probs = np.concatenate([all_preds_probs, pred_probs.detach().cpu().numpy()[:, 1]])

        total_cnf_matrix += confusion_matrix(labels.cpu().numpy(), pred_classes.cpu().numpy(), labels=[0, 1])
        total_acc, acc, sensitivity, specificity, precision, f1 = pre_cal(total_cnf_matrix)

        loss = loss_function(pred_probs, labels)
        val_loss.append(loss.item())
        val_loss_avg = np.sum(val_loss) / iter_num

        postfix = OrderedDict([
            ('val_loss', val_loss_avg),
            ('val_total_acc', total_acc),
            ('val_acc', acc),
            ('val_sensitivity', sensitivity),
            ('val_specificity', specificity),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    # compute ROC curve and AUC value
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds_probs, pos_label=1, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    # plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # save ROC curve
    os.makedirs('./results/{}/roc_fig'.format(args.name), exist_ok=True)
    plt.savefig('./results/{}/roc_fig/roc-epoch-{}.png'.format(args.name, epoch))
    # plt.show()

    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })

    # save ROC data
    os.makedirs('./results/{}/roc_data'.format(args.name), exist_ok=True)
    roc_data.to_csv('./results/{}/roc_data/roc_data-epoch-{}.csv'.format(args.name, epoch), index=False)

    return OrderedDict([('val_loss', val_loss_avg),
                        ('val_total_acc', total_acc),
                        ('val_acc', acc),
                        ('val_sensitivity', sensitivity),
                        ('val_specificity', specificity),
                        ('val_precision', precision),
                        ('val_f1', f1),
                        ('val_auc', roc_auc),
                        ])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='new_model')
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--n_input_channels', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)

    # dataset
    parser.add_argument('--dataset', default='lesions_classification_dataset', help='brats2019, lesions_classification_dataset')
    parser.add_argument('--dataset_folder', default="./dataset")

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'AdamW'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str, help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'LambdaLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    args = parser.parse_args()

    main(args)


