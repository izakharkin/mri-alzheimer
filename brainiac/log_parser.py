import re
import numpy as np
import itertools as it
from operator import itemgetter
ig0, ig1 = map(itemgetter, (0, 1))


def parser_log(string, log_type):
    if log_type == 'train_loss':
        pattern = re.compile(r'(^[0-9\-\s\:\,]+) - root - INFO - Train Epoch: (?P<epoch>\d+) \| Iter: \[(?P<iter>\d+) / (?P<max_iter>\d+) \(\d+%\)\] \| Loss: (?P<loss>[\d\.]+)\n')
        found = pattern.match(string)
        if found is not None:
            res = found.groupdict()
            res['epoch'], res['iter'], res['max_iter'] = map(int, (res['epoch'], res['iter'], res['max_iter']))
            res['loss'] = float(res['loss'])
            return res
        else:
            return None
    elif log_type == 'train_acc':
        pattern = re.compile(r'(^[0-9\-\s\:\,]+) - root - INFO - Train Epoch: (?P<epoch>\d+) \| Accuracy: (?P<correct>\d+)/(?P<all>\d+) \((?P<acc>0.\d+)\) \| RocAuc: (?P<roc_auc>0.\d+)\n')
        found = pattern.match(string)
        if found is not None:
            res = found.groupdict()
            res['epoch'], res['correct'], res['all'] = map(int, (res['epoch'], res['correct'], res['all']))
            res['acc'], res['roc_auc'] = map(float, (res['acc'], res['roc_auc']))
            return res
        else:
            return None
    elif log_type == 'test':
        'Test Epoch: {:04d} |  Accuracy: {}/{} ({:.3f}) | RocAuc: {:.3f}'
        pattern = re.compile(r'(^[0-9\-\s\:\,]+) - root - INFO - Test Epoch: (?P<epoch>\d+) \| Average loss: (?P<loss>[\d\.]+) \| Accuracy: (?P<correct>\d+)/(?P<all>\d+) \((?P<acc>0.\d+)\) \| RocAuc: (?P<roc_auc>0.\d+)\n')
        found = pattern.match(string)
        if found is not None:
            res = found.groupdict()
            res['correct'], res['all'] = map(int, (res['correct'], res['all']))
            res['loss'], res['acc'], res['roc_auc'] = map(float, (res['loss'], res['acc'], res['roc_auc']))
            return res
        else:
            return None
    else:
        raise NotImplementedError
        
def parser_all_log(path):
    train_loss_all = []
    train_loss_epoch = []
    test_loss = []
    
    train_acc = []
    test_acc = []
    
    train_roc_auc = []
    test_roc_auc = []
    
    f = open(path)
    for s in f.readlines():
        res = parser_log(s, 'train_loss')
        if res is not None:
            train_loss_all.append(res['loss'])
            train_loss_epoch.append((res['epoch'] , res['loss']))
            continue
        
        res = parser_log(s, 'train_acc')
        if res is not None:
            train_acc.append(res['acc'])
            train_roc_auc.append(res['roc_auc'])
            continue
        
        res = parser_log(s, 'test')
        if res is not None:
            test_loss.append(res['loss'])
            test_acc.append(res['acc'])
            test_roc_auc.append(res['roc_auc'])
            continue
        
    train_loss_epoch = [np.mean(list(map(ig1, val))) for i , val in it.groupby(train_loss_epoch, ig0)]
    return train_loss_all, train_loss_epoch, test_loss, train_acc, test_acc, train_roc_auc, test_roc_auc

#Version for old type of logs

def old_parser_log(string, log_type):
    if log_type == 'train_loss':
        pattern = re.compile(r'(^[0-9\-\s\:\,]+) - root - INFO - Train Epoch: (?P<epoch>\d+)  \[(?P<iter>\d+) / (?P<max_iter>\d+) \(\d+%\)\]\tLoss: (?P<loss>[\d\.]+)\n')
        found = pattern.match(string)
        if found is not None:
            res = found.groupdict()
            res['epoch'], res['iter'], res['max_iter'] = map(int, (res['epoch'], res['iter'], res['max_iter']))
            res['loss'] = float(res['loss'])
            return res
        else:
            return None
    elif log_type == 'train_acc':
        pattern = re.compile(r'(^[0-9\-\s\:\,]+) - root - INFO - Train Epoch: (?P<epoch>\d+) Accuracy: (?P<correct>\d+)/(?P<all>\d+) \(\d+\%\)\n')
        found = pattern.match(string)
        if found is not None:
            res = found.groupdict()
            res['epoch'], res['correct'], res['all'] = map(int, (res['epoch'], res['correct'], res['all']))
            return res
        else:
            return None
    elif log_type == 'test':
        pattern = re.compile(r'(^[0-9\-\s\:\,]+) - root - INFO - Test set: Average loss: (?P<loss>\d+.\d+), Accuracy: (?P<correct>\d+)/(?P<all>\d+) \(\d+\%\)\n')
        found = pattern.match(string)
        if found is not None:
            res = found.groupdict()
            res['loss'] = float(res['loss'])
            res['correct'], res['all'] = map(int, (res['correct'], res['all']))
            return res
        else:
            return None
    else:
        raise NotImplementedError
        
def old_parser_all_log(path):
    train_loss_all = []
    train_loss_epoch = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    f = open(path)
    for s in f.readlines():
        res = old_parser_log(s, 'train_loss')
        if res is not None:
            train_loss_all.append(res['loss'])
            train_loss_epoch.append((res['epoch'] , res['loss']))
            continue
        
        res = old_parser_log(s, 'train_acc')
        if res is not None:
            train_acc.append(res['correct'] / res['all'])
            continue
        
        res = old_parser_log(s, 'test')
        if res is not None:
            test_loss.append(res['loss'])
            test_acc.append(res['correct'] / res['all'])
            continue
        
    train_loss_epoch = [np.mean(list(map(ig1, val))) for i , val in it.groupby(train_loss_epoch, ig0)]
    return train_loss_all, train_loss_epoch, test_loss, train_acc, test_acc