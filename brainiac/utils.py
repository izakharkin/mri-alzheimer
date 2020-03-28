import os
import torch
from torch.optim import Adam, RMSprop, SGD

def get_optimizer(model, args):
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def save_model_epoch(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))

    
def load_model(model, path):
    params = torch.load(path)
    model.load_state_dict(params['model_state_dict'])
    return model, params['epoch']