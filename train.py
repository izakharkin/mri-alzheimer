import argparse

import os
import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from brainiac.utils import *
from brainiac.train_utils import *
from brainiac.log_helper import *

import logging

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
                        required=True, help='Network archetecture')
    parser.add_argument('--data_path', type=str,
                        default='../ADNI-processed', help='Full path to a data .csv file')
    parser.add_argument('--classes', type=str,
                        default="['CN', 'AD']", help='Classes for experiment')
    
    parser.add_argument('--use_regression', type=str2bool,
                       default=False, help='Train regression model')
    
    parser.add_argument('--num_epoch', type=int,
                        default=200, help='Number of epoch')
    parser.add_argument('--batch_size', type=int,
                        default=4, help='Batch size')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='Optimizer',
                        choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--lr', type=float,
                        default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-3, help='Weight decay')

    parser.add_argument('--use_augmentation', type=str2bool,
                        default=True, help='Use or not augmentation')
    parser.add_argument('--use_sampling', type=str2bool,
                        default=True, help='Use sampling or not')
    parser.add_argument('--sampling_type', type=str,
                        default='over', help='Type of sampling (over and under)')
    
    parser.add_argument('--train_print_every', type=int,
                        default=1, help='Interval of logging in train')
    parser.add_argument('--test_print_every', type=int,
                        default=1, help='Interval of logging in test')
    
    parser.add_argument('--use_scheduler', type=str2bool,
                        default=True, help='Use scheduler or not')
    parser.add_argument('--scheduler_step', type=str,
                        default='[50, 100, 150]', help='Scheduler\'s steps')
    parser.add_argument('--scheduler_gamma', type=float,
                        default=0.1, help='Scheduler\'s gamma')
    
    parser.add_argument('--device', type=str,
                        default='cuda', help='Computing device')
    
    parser.add_argument('--use_pretrain', type=str2bool,
                        default=False, help='Use pretrain model')
    parser.add_argument('--path_pretrain', type=str,
                        default=None, help='Path to pretrain model')
    parser.add_argument('--pretrain_head', type=int,
                        default=2, help='Num classes of pretrain model')
    
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    save_dir = 'trained_model/{}/{}-classes-{}_optim-{}_aug-{}_sampling-{}_lr-{}_scheduler-{}_pretrain-{}/'.format(
        args.model, args.data_path, '-'.join([str(i) for i in eval(args.classes)]),
        args.optimizer, int(args.use_augmentation), int(args.use_sampling),
        args.lr, int(args.use_scheduler), int(args.use_sampling))
    args.save_dir = save_dir
    
    args.images_path = args.data_path + '/images/'
    args.data_path = args.data_path + '/data.csv'
    
    if args.device < 'cpu':
        args.device = 'cuda:' + args.device
    
    return args    

def train_epoch(epoch, model, data_loader, optimizer, args):
    model.train()
    y_true = []
    y_pred = []
    
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(args.device))
        loss = F.cross_entropy(output, target.to(args.device))
        loss.backward()
        optimizer.step()
        
        y_true.append(target.detach().cpu().numpy())
        y_pred.append(output.detach().cpu().numpy())
        
        if batch_idx % args.train_print_every == 0:
            logging.info('Train Epoch: {:04d} | Iter: [{:04d} / {:04d} ({:02d}%)] | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), data_loader.sampler.num_samples,
                int(100. * batch_idx / len(data_loader)), loss.item()))
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    num_samples = len(y_true)
    correct, acc, roc_auc = get_metrics(y_true, y_pred)

    
    logging.info('Train Epoch: {:04d} | Accuracy: {}/{} ({:.3f}) | RocAuc: {:.3f}'.format(
        epoch, correct, num_samples, acc, roc_auc))

def test_epoch(epoch, model, data_loader, args):
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(args.device))
            test_loss += F.cross_entropy(output, target.to(args.device), reduction='sum').item() # sum up batch loss
            
            y_true.append(target.cpu().numpy())
            y_pred.append(output.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    num_samples = len(y_true)
    correct, acc, roc_auc = get_metrics(y_true, y_pred)
    test_loss /= num_samples
    
    logging.info('Test Epoch: {:04d} | Average loss: {:.4f} | Accuracy: {}/{} ({:.3f}) | RocAuc: {:.3f}'.format(  
        epoch, test_loss, correct, num_samples, acc, roc_auc))
    return acc
    

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)
    
    logging.info('-'*20 + 'Data' + '-'*20)
    train_loader, test_loader, num_classes = make_data(args)
    
    logging.info('-'*20 + 'Model' + '-'*20)
    model, init_epoch = make_model(args, num_classes)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    logging.info(model)
    
    logging.info('-'*20 + 'Train' + '-'*20)    
    model.to(args.device)
    best_acc = 0
    best_epoch = -1
    for epoch in range(init_epoch, init_epoch + args.num_epoch):
        train_epoch(epoch, model, train_loader, optimizer, args)
        
        if scheduler:
            scheduler.step()
        
        logging.info('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        if epoch % args.test_print_every == 0:
            current_acc = test_epoch(epoch, model, test_loader, args)
            if current_acc > best_acc:
                save_model_epoch(model, args.save_dir, epoch, best_epoch)
                best_acc, best_epoch = current_acc, epoch
                logging.info('Save model at {} epoch'.format(epoch))
        
    save_model_epoch(model, args.save_dir, epoch)
        
if __name__ == '__main__':
    args = parse_args()
    train(args)
