import logging
import pandas as pd
import numpy as np

from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import WeightedRandomSampler, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

from .models import *
from .loader import *
from .utils import *


def get_metrics(y_true, pred):
    y_prob = np.exp(pred)
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    y_pred = np.argmax(pred, axis=1)
    
    if y_prob.shape[1] == 2: #Binary classification
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else: #Balanced accuracy
        roc_auc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    correct = (y_true == y_pred).sum()
    
    return correct, acc, roc_auc

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

def make_data(args):
    
    classes = eval(args.classes)
    num_classes = len(classes)
    df = pd.read_csv(args.data_path)
    df_train, df_test = train_test_split_df(df, classes)
    
    logging.info(('\t{}'*num_classes).format(*classes))
    logging.info(('Train' + '\t{}'*num_classes).format(*df_train['Group'].value_counts()[np.arange(num_classes)].values))
    logging.info(('Test' + '\t{}'*num_classes).format(*df_test['Group'].value_counts()[np.arange(num_classes)].values))
    
    sampler = None
    if args.use_sampling:
        labels = df_train['Group'].values
        sampler = make_sampler(labels, args.sampling_type)
    if args.use_regression:
        levels = np.linspace(0, 100, num_classes + 1)
        train_dataset = ADNIRegressionDataset(df_train, train=args.use_augmentation, images_path=args.images_path, levels=levels)
        test_dataset = ADNIRegressionDataset(df_test, train=False, images_path=args.images_path, levels=levels)
    else:
        train_dataset = ADNIClassificationDataset(df_train, train=args.use_augmentation, images_path=args.images_path)
        test_dataset = ADNIClassificationDataset(df_test, train=False, images_path=args.images_path)
    shuffle = True if sampler is None else False
    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    return train_loader, test_loader, num_classes


def choose_model(args, num_classes):
    if args.model == 'CNN':
        model = SimpleCNN(num_classes=num_classes)
    elif args.model == 'AlexNet3D':
        model = AlexNet3D(num_classes=num_classes)
    elif args.model == 'AlexNet2D':
        model = AlexNet2D(num_classes=num_classes)
    elif args.model == 'LeNet3D':
        model = LeNet3D(num_classes=num_classes)
    elif args.model == 'ResNet50':
        model = resnet50(num_classes=num_classes)
    elif args.model == 'ResNet152':
        model = resnet152(num_classes=num_classes)
    else:
        raise NotImplementedError
    return model


def make_model(args, num_classes):
    if args.use_pretrain:
        assert args.path_pretrain is not None
        model = choose_model(args, args.pretrain_head)
        model, init_epoch = load_model(model, args.path_pretrain)
        logging.info('Load model from {}'.format(args.path_pretrain))
    
        if args.pretrain_head != num_classes:
            init_epoch = 0
            if isinstance(model, ResNet):
                model = resnet_change_head(model, num_classes)
            else:
                raise NotImplementedError
        
        return model, init_epoch
    else:
        model = choose_model(args, num_classes)
        return model, 0

def get_optimizer(model, args):
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer    
    
def get_scheduler(optimizer, args):
    if args.use_scheduler:
        scheduler = MultiStepLR(optimizer, gamma=args.scheduler_gamma, milestones=eval(args.scheduler_step))
    else:
        scheduler = None
    return scheduler

def get_pred_with_levels(outputs, mid_levels, args):
    pred = torch.zeros(len(outputs)).to(args.device)
    for i, output in enumerate(outputs):
        pred[i] = torch.argmin(torch.abs(mid_levels - output))
    return pred