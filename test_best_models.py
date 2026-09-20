# -*- coding: utf-8 -*-
"""
独立脚本：加载5折最佳checkpoint，在对应测试集上评估，汇总输出均值和方差。
不修改任何现有文件。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

from model import SSGraphDTI, collate_fn
from dataset import CustomDataSet
from hyperparameter import hyperparameter

warnings.filterwarnings("ignore")

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATASET = "DrugBank35022"
K_FOLD = 5

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]
    return trainset, validset

def evaluate(model, dataloader, loss_fn):
    model.eval()
    Y, P, S, losses = [], [], [], []
    with torch.no_grad():
        for _, data in tqdm(enumerate(BackgroundGenerator(dataloader)), total=len(dataloader)):
            drugIDs, proIDs, numSMILESs, numFASTAs, labels = data
            numSMILESs, numFASTAs, labels = numSMILESs.cuda(), numFASTAs.cuda(), labels.cuda()
            scores = model(drugIDs, proIDs, numSMILESs, numFASTAs)
            losses.append(loss_fn(scores, labels).item())
            labels_np = labels.cpu().numpy()
            scores_np = F.softmax(scores, 1).cpu().numpy()
            preds = np.argmax(scores_np, axis=1)
            Y.extend(labels_np)
            P.extend(preds)
            S.extend(scores_np[:, 1])
    acc = accuracy_score(Y, P)
    prec = precision_score(Y, P)
    rec = recall_score(Y, P)
    auc_val = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    prc = auc(fpr, tpr)
    return acc, prec, rec, auc_val, prc

if __name__ == "__main__":
    hp = hyperparameter()
    Loss = nn.CrossEntropyLoss()

    dir_input = './data/{}.txt'.format(DATASET)
    with open(dir_input, "r") as f:
        train_data_list = f.read().strip().split('\n')
    dataset = shuffle_dataset(train_data_list, SEED)

    Accuracy_List, Precision_List, Recall_List, AUC_List, PRC_List = [], [], [], [], []

    for i_fold in range(K_FOLD):
        print('*' * 25, 'Fold', i_fold + 1, '*' * 25)

        # 复现与训练时完全相同的数据划分
        train_dataset, test_dataset = get_kfold_data(i_fold, dataset)
        test_dataset_load = DataLoader(
            CustomDataSet(test_dataset),
            batch_size=hp.Batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
        )

        # 加载该折最佳权重
        ckpt_path = './{}/{}/valid_best_checkpoint.pth'.format(DATASET, i_fold)
        model = SSGraphDTI(hp).cuda()
        model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
        print(f'Loaded checkpoint: {ckpt_path}')

        acc, prec, rec, auc_val, prc = evaluate(model, test_dataset_load, Loss)
        print(f'  Accuracy:{acc:.4f}  Precision:{prec:.4f}  Recall:{rec:.4f}  AUC:{auc_val:.4f}  PRC:{prc:.4f}')

        Accuracy_List.append(acc)
        Precision_List.append(prec)
        Recall_List.append(rec)
        AUC_List.append(auc_val)
        PRC_List.append(prc)

    # 汇总
    print('\n' + '=' * 60)
    print('5-Fold Best Model Results (mean ± var):')
    for name, lst in [('Accuracy', Accuracy_List), ('Precision', Precision_List),
                      ('Recall', Recall_List), ('AUC', AUC_List), ('PRC', PRC_List)]:
        print(f'  {name}(var): {np.mean(lst):.4f}({np.var(lst):.4f})')

    F1_list = [2*p*r/(p+r) for p, r in zip(Precision_List, Recall_List)]
    print(f'  F1_score(var): {np.mean(F1_list):.4f}({np.var(F1_list):.4f})')
