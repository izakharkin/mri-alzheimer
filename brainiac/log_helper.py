import os
import logging
import csv
from collections import OrderedDict
import re

import numpy as np
import itertools as it
from operator import itemgetter
ig0, ig1 = map(itemgetter, (0, 1))



def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):

    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All logs will be saved to %s" %logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder

def parser_log(string, log_type):
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
        
def parser_all_log(path):
    train_loss_all = []
    train_loss_epoch = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    f = open(path)
    for s in f.readlines():
        res = parser_log(s, 'train_loss')
        if res is not None:
            train_loss_all.append(res['loss'])
            train_loss_epoch.append((res['epoch'] , res['loss']))
            continue
        
        res = parser_log(s, 'train_acc')
        if res is not None:
            train_acc.append(res['correct'] / res['all'])
            continue
        
        res = parser_log(s, 'test')
        if res is not None:
            test_loss.append(res['loss'])
            test_acc.append(res['correct'] / res['all'])
            continue
        
    train_loss_epoch = [np.mean(list(map(ig1, val))) for i , val in it.groupby(train_loss_epoch, ig0)]
    return train_loss_all, train_loss_epoch, train_acc, test_loss, test_acc