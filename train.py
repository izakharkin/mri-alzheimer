import argparse

import os
import random
import pandas as pd
import numpy as np
import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from brainiac.models import *
from brainiac.loader import *
from brainiac.utils import *
from brainiac.log_helper import *

import logging


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
                        required=True, help='Network archetecture')
    parser.add_argument('--data_path', type=str,
                        default='/home/ADNI-processed/data.csv', help='Full path to a data .csv file')
    parser.add_argument('--classes', type=str,
                        default="['CN', 'EMCI', 'MCI', 'LMCI', 'AD']", help='Classes for experiment')
#     ['CN', 'SMC', 'EMCI', 'MCI', 'LMCI', 'AD']
    
    parser.add_argument('--num_epoch', type=int,
                        default=200, help='Number of epoch')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='Batch size')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='Optimizer',
                        choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--lr', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001, help='Weight decay')

    parser.add_argument('--use_augmentation', type=bool,
                        default=True, help='Use or not augmentation')
    parser.add_argument('--use_sampling', type=bool,
                        default=True, help='Use sampling or not')
    parser.add_argument('--sampling_type', type=str,
                        default='over', help='Type of sampling (over and under)')

    parser.add_argument('--train_print_every', type=int,
                        default=1, help='Interval of logging in train')
    parser.add_argument('--test_print_every', type=int,
                        default=1, help='Interval of logging in test')
    
    parser.add_argument('--use_scheduler', type=bool,
                        default=False, help='Use scheduler or not')
    parser.add_argument('--scheduler_step', type=str,
                        default='[50, 100, 150]', help='Scheduler\'s steps')
    parser.add_argument('--scheduler_gamma', type=float,
                        default=0.1, help='Scheduler\'s gamma')
    
    parser.add_argument('--device', type=str,
                        default='cuda', help='Computing device', 
                        choices=['cpu', 'cuda'])
    
    parser.add_argument('--use_pretrain', type=bool,
                        default=False, help='Use pretrain model')
    parser.add_argument('--path_pretrain', type=str,
                        default=None, help='Path to pretrain model')

    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    save_dir = 'trained_model/{}/classes-{}_optim-{}_aug-{}_sampling-{}_lr-{}_scheduler-{}_pretrain-{}/'.format(
        args.model, '-'.join([str(i) for i in eval(args.classes)]),
        args.optimizer, int(args.use_augmentation), int(args.use_sampling),
        args.lr, int(args.use_scheduler), int(args.use_sampling))
    args.save_dir = save_dir
    
    return args



def get_pred(outputs, mid_levels, args):
    pred = torch.zeros(len(outputs)).to(args.device)
    for i, output in enumerate(outputs):
        pred[i] = torch.argmin(torch.abs(mid_levels - output))
    return pred


def train_epoch(epoch, model, data_loader, optimizer, args, mid_levels):
    model.train()
    correct = 0
    num_samples = 0
    for batch_idx, (data, target, real) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(args.device)) # from 0 to 1
#         loss = F.cross_entropy(output, target.to(args.device))
        target_tmp = target.to(args.device)[:, None]
        loss = F.mse_loss(output, target_tmp)
        pred = get_pred(output, mid_levels, args)
        if batch_idx % 10 == 0:
            print(f'output: {output.view(-1)}, pred: {pred}, real: {real}')
        correct += pred.eq(real.to(args.device)).sum().item()
        num_samples += len(target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.train_print_every == 0:
            logging.info('Train Epoch: {:04d}  [{:04d} / {:04d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), data_loader.sampler.num_samples,
                100. * batch_idx / len(data_loader), loss.item()))
    logging.info('Train Epoch: {:04d} Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, correct, num_samples, 100. * correct / num_samples))

def test_epoch(model, args, data_loader, name_set, mid_levels):
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, target, real in data_loader:
            output = model(data.to(args.device))
#             test_loss += F.cross_entropy(output, target.to(args.device), reduction='sum').item() # sum up batch loss
            target_tmp = target.to(args.device)[:, None]
            test_loss = F.mse_loss(output, target_tmp).sum()
            pred = get_pred(output, mid_levels, args)
            correct += pred.eq(real.to(args.device)).sum().item()
            num_samples += len(target)

    test_loss /= num_samples  # len(data_loader.dataset)
    logging.info('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        name_set, test_loss, correct, num_samples,
        100. * correct / num_samples))
    return correct / num_samples

def make_sampler(labels, mode='over'):
    class_sample_count = np.unique(labels, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[labels]
    n_classes = len(class_sample_count)
    if mode == 'over':
        num_samples = max(class_sample_count) * n_classes
    elif mode == 'under':
        num_samples = min(class_sample_count) * n_classes
    else:
        raise NotImplementedError

    sampler = WeightedRandomSampler(samples_weight, int(num_samples), replacement=True)
    return sampler


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)
    
    logging.info('-'*20 + 'Data' + '-'*20)
    
    classes = eval(args.classes)
    n_classes = len(classes)
    df = pd.read_csv(args.data_path)
    df_train, df_test, levels = train_test_split_df(df, classes)
    mid_levels = torch.tensor([(levels[i] + levels[i-1])/2. for i in range(1, len(levels))]).to(args.device)

    logging.info(('\t{}'*n_classes).format(*classes))
    logging.info(('Train' + '\t{}'*n_classes).format(*df_train['Group'].value_counts()[np.arange(n_classes)].values))
    logging.info(('Test' + '\t{}'*n_classes).format(*df_test['Group'].value_counts()[np.arange(n_classes)].values))

    sampler = None
    if args.use_sampling:
        labels = df_train['Group'].values
        sampler = make_sampler(labels, args.sampling_type)

    train_dataset = ADNIClassificationDataset(df_train, train=args.use_augmentation, levels=levels)
    test_dataset = ADNIClassificationDataset(df_test, train=False, levels=levels)

    shuffle = True if sampler is None else False
    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    logging.info('-'*20 + 'Model' + '-'*20)
    if args.model == 'CNN':
        model = SimpleCNN(num_classes=n_classes)
    elif args.model == 'AlexNet3D':
        model = AlexNet3D(num_classes=n_classes)
    elif args.model == 'AlexNet2D':
        model = AlexNet2D(num_classes=n_classes)
    elif args.model == 'LeNet3D':
        model = LeNet3D(num_classes=n_classes)
    elif args.model == 'ResNet3D':
        model = ResNet3D(num_classes=n_classes)
    else:
        raise NotImplementedError

    optimizer = get_optimizer(model, args)

    scheduler = None
    if args.use_scheduler:
        scheduler = MultiStepLR(optimizer, gamma=args.scheduler_gamma,
                               milestones=eval(args.scheduler_step))

    logging.info(model)
    logging.info('-'*20 + 'Train' + '-'*20)
    
    init_epoch = 0
    if args.use_pretrain:
        assert args.path_pretrain is not None
        model, init_epoch = load_model(model, args.path_pretrain)
        logging.info('Load model from {}. And start at {} epoch'.format(args.path_pretrain, init_epoch))
        
    model.to(args.device)
    best_acc = 0
    best_epoch = -1
    for epoch in range(init_epoch, args.num_epoch):
        
        train_epoch(epoch, model, train_loader, optimizer, args, mid_levels)
        
        if scheduler:
            scheduler.step()

        logging.info('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        if epoch % args.test_print_every == 0:
            current_acc = test_epoch(model, args, test_loader, 'Test', mid_levels)
            if current_acc > best_acc:
                save_model_epoch(model, args.save_dir, epoch, best_epoch)
                best_acc, best_epoch = current_acc, epoch
                logging.info('Save model at {} epoch'.format(epoch))
        
    save_model_epoch(model, args.save_dir, epoch)
        
if __name__ == '__main__':
    args = parse_args()
    train(args)
