# from __future__ import division
# from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.optim as optim
from utils import generator, accuracy
import config
from models import *
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr
from preprocess import get_fc_graph
from index import generate_train_test_indices
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Training settings
parser = argparse.ArgumentParser(description='Process hyper-parameters')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=7, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.00015, #0.015
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size.')
parser.add_argument('--model', type=str, default='GCN',
                    help='Model type.')
parser.add_argument('--task', type=str, default='age_predict',
                    help='Task type.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def sex_predict_criterion(pred, target):
    pred = pred.float()  # Convert predictions to Float
    target = target.float()
    BCE = nn.BCELoss()
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    return BCE(pred, target)  # Use Binary Cross-Entropy loss

def age_predict_criterion(pred, target):
    BCE = nn.BCELoss()
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    return MSE(pred, target)  # Use Mean Squared Error for age prediction

# Function to collect node features after the first GCN processing

def brain_batch_train(model, task, criterion, fc, sc, features, label, optimizer):
    # Train the model on a batch
    model.train()
    optimizer.zero_grad()
    B = fc.shape[0]
    if args.model == 'MLP':
        inp = features.view(B, -1).cuda()
        train_output = model(inp, task)
    elif args.model == 'MLP_160':
        fea = []
        for i in range(B):
            fea.append(np.concatenate([np.diag(feature[i, :80]), np.diag(feature[i, 80:])]))
        feature = np.stack(fea)
        feature = torch.tensor(feature)
        inp = feature.view(B, -1).cuda()
        train_output = model(inp, task)
    else:
        train_output = model(features.cuda(), sc.cuda(), task)
    loss_train = criterion(train_output, label.cuda())
    acc_train = accuracy(train_output.cpu(), label)
    loss_train.backward()
    optimizer.step()
    return loss_train, acc_train, len(train_output)

def brain_batch_test(model, task, criterion, fc, sc, features, label, predp, gt):
    # Test the model on a batch
    model.eval()
    B = fc.shape[0]
    with torch.no_grad():
        if args.model == 'MLP':
            inp = features.view(B, -1).cuda()
            test_output = model(inp, task)
        elif args.model == 'MLP_160':
            fea = []
            for i in range(B):
                fea.append(np.concatenate([np.diag(feature[i, :80]), np.diag(feature[i, 80:])]))
            feature = np.stack(fea)
            feature = torch.tensor(feature)
            inp = feature.view(B, -1).cuda()
            test_output = model(inp, task)
        else:
            test_output = model(features.cuda(), sc.cuda(), task)
        loss_test = criterion(test_output, label.cuda())
        acc_test = accuracy(test_output.cpu(), label)
        predp.extend(test_output.cpu())
        gt.extend(label)
    return loss_test, acc_test, len(test_output), predp, gt

def brain_train():
    num_samples = 196
    train_idx, test_idx = generate_train_test_indices(num_samples)
    best_acc_5fold, best_acc_auc_5fold, best_auc_5fold, best_auc_acc_5fold = [], [], [], []
    best_std_5fold, best_mae_5fold, best_mae_pearson_5fold, best_pearson_5fold, best_pearson_mae_5fold = [], [], [], [], []
    for i in range(5):
        # Set random seed for reproducibility
        task = args.task
        best_acc, best_acc_auc = 0, 0
        best_auc, best_auc_acc = 0, 0
        best_mae, best_mae_pearson = 1000, 0
        best_pearson, best_pearson_mae = 0, 1000

        # Select criterion based on the task type
        if task == 'sex_predict':
            criterion = sex_predict_criterion
        elif task == 'age_predict':
            criterion = age_predict_criterion

        # Select the model type
        if args.model == 'GCN':
            net = GCN(188, 120, 1, args.dropout)
        elif args.model == 'MLP':
            net = MLP(188 * 188, 1, args.dropout)
        elif args.model == 'GraphSAGE':
            net = GraphSAGE(188, 120, 1, args.dropout)
        net.cuda()
        optimizer = optim.SGD(net.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

        metrics_history = {
            'loss_train': [],
            'acc_train': [],
            'loss_test': [],
            'acc_test': [],
            'auc_test': [],
            'std_test': [],
            'mae_test': [],
            'pearson_test': []
        }
        print('==================================== fold:{} ============================================'.format(i+1))

        # Load data
        fc_list, sc_list = [], []
        fc_folder = './dataset/raw_data/fc_adjacency/'
        sc_folder = './dataset/raw_data/sc_adjacency/'
        list_path = './dataset/raw_data/control_subject_list.csv'
        fc_list, sc_list = get_fc_graph(fc_folder, sc_folder, list_path)
        train_generator, test_generator = generator(train_idx, test_idx, args.batch_size, task, fc_list, sc_list)

        for epoch in range(args.epochs):
            st = time.time()
            loss_train_list, acc_train_list, len_train_list = [], [], []
            loss_test_list, acc_test_list, len_test_list = [], [], []
            predpall, gtall = [], []

            for fc_train, sc_train, features, label_train in train_generator:
                loss_train, acc_train, len_train = brain_batch_train(net, task, criterion, fc_train, sc_train, features, label_train, optimizer)
                loss_train_list.append(loss_train * len_train)
                acc_train_list.append(acc_train * len_train)
                len_train_list.append(len_train)

            for fc_test, sc_test, features_test, label_test in test_generator:
                loss_test, acc_test, len_test, predpall, gtall = brain_batch_test(net, task, criterion, fc_test, sc_test, features_test, label_test, predpall, gtall)
                loss_test_list.append(loss_test*len_test)
                acc_test_list.append(acc_test*len_test)
                len_test_list.append(len_test)

            predpall = np.array([p.item() for p in predpall])
            gtall = np.array([g.item() for g in gtall])

            if task == 'age_predict':
                std = np.std(gtall - predpall)
                mae = mean_absolute_error(gtall, predpall)
                pearson = pearsonr(gtall, predpall)[0]

                metrics_history['std_test'].append(std)
                metrics_history['mae_test'].append(mae)
                metrics_history['pearson_test'].append(pearson)

                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(sum(loss_train_list) / sum(len_train_list)),
                      'loss_test: {:.4f}'.format(sum(loss_test_list) / sum(len_test_list)),
                      'std_test: {:.4f}'.format(std),
                      'mea_test: {:.4f}'.format(mae),
                      'pearson_test: {:.4f}'.format(pearson),
                      'time: {:.4f}s'.format(time.time() - st))
                if mae < best_mae:
                    best_std = std
                    best_mae = mae
                    best_mae_pearson = pearson
                    torch.save(net.state_dict(),
                               './best_models/' + args.model + '_net_mae_' + task + '_' + str(i) + '.pth')
                if pearson > best_pearson:
                    best_pearson = pearson
                    best_pearson_mae = mae
                    torch.save(net.state_dict(),
                               './best_models/' + args.model + '_net_pearson_' + task + '_' + str(i) + '.pth')
        best_acc_5fold.append(best_acc)
        best_auc_5fold.append(best_auc)
        best_acc_auc_5fold.append(best_acc_auc)
        best_auc_acc_5fold.append(best_auc_acc)
        best_std_5fold.append(best_std)
        best_mae_5fold.append(best_mae)
        best_mae_pearson_5fold.append(best_mae_pearson)
        best_pearson_5fold.append(best_pearson)
        best_pearson_mae_5fold.append(best_pearson_mae)

    if task == 'age_predict':
        d = {'best_std_5fold': best_std_5fold,
             'best_mae_5fold': best_mae_5fold,
             'best_mae_pearson_5fold': best_mae_pearson_5fold,
             'best_pearson_5fold': best_pearson_5fold,
             'best_pearson_mae_5fold': best_pearson_mae_5fold}
        df = pd.DataFrame(data=d)
        df.to_csv('./results/result_' + args.model + task + '_.csv', index=False)

        print('best_mae_5fold is:', best_mae_5fold, 'mean mae of them is:', sum(best_mae_5fold) / len(best_mae_5fold),
              "mean pearson of them is:", sum(best_mae_pearson_5fold) / len(best_mae_pearson_5fold))
        print('best_pearson_5fold is:', best_pearson_5fold, 'mean pearson of them is:',
              sum(best_pearson_5fold) / len(best_pearson_5fold),
              "mean mae of them is:", sum(best_pearson_mae_5fold) / len(best_pearson_mae_5fold))
        print('best_std_5fold is:', best_std_5fold, 'mean std of them is:', sum(best_std_5fold) / len(best_std_5fold))

        plt.figure(figsize=(10, 5))
        plt.plot(metrics_history['std_test'], marker='o', linestyle='-', color='r')
        plt.title('Standard Deviation of Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Standard Deviation')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(metrics_history['mae_test'], marker='o', linestyle='-', color='g')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(metrics_history['pearson_test'], marker='o', linestyle='-', color='m')
        plt.title('Pearson Correlation Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Pearson')
        plt.show()

if __name__ == '__main__':
    print(args.model + '_' + args.task)
    brain_train()
