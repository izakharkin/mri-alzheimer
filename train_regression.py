import logging
import random

from sklearn.metrics import f1_score

from brainiac.log_helper import *
from brainiac.train_utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        required=True, help='Network archetecture')
    parser.add_argument('--data_path', type=str,
                        default='D:/home/ADNI_cut/data.csv', help='Full path to a data .csv file')
    parser.add_argument('--images_path', type=str,
                        default='D:/home/ADNI_cut/', help='Full path to a data .csv file')
    parser.add_argument('--classes', type=str,
                        default="['CN', 'EMCI', 'MCI', 'LMCI', 'AD']", help='Classes for experiment')

    parser.add_argument('--num_epoch', type=int,
                        default=200, help='Number of epoch')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='Batch size')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='Optimizer',
                        choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--lr', type=float,
                        default=3e-05, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay')

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
                        default=True, help='Use pretrain model')
    parser.add_argument('--path_pretrain', type=str,
                        default='C:/Users/User/PycharmProjects/mri-alzheimer/trained_model/pretrained/model_epoch72.pth',
                        help='Path to pretrain model')
    parser.add_argument('--pretrain_head', type=int,
                        default=2, help='Num classes of pretrain model')
    parser.add_argument('--model_num_classes', type=int,
                        default=1, help='Num classes to return of model')

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--use_regression', type=str2bool,
                        default=True, help='If regression is used')

    args = parser.parse_args()

    save_dir = 'trained_model/{}/classes-{}_optim-{}_aug-{}_sampling-{}_lr-{}_scheduler-{}_pretrain-{}/'.format(
        args.model, '-'.join([str(i) for i in eval(args.classes)]),
        args.optimizer, int(args.use_augmentation), int(args.use_sampling),
        args.lr, int(args.use_scheduler), int(args.use_sampling))
    args.save_dir = save_dir

    if args.device < 'cpu':
        args.device = 'cuda:' + args.device

    return args


def train_epoch(epoch, model, data_loader, optimizer, args, mid_levels):
    model.train()
    correct = 0
    num_samples = 0
    y_true = []
    y_pred = []

    for batch_idx, (data, target, real) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(args.device))  # from 0 to 1
        #         loss = F.cross_entropy(output, target.to(args.device))
        target_tmp = target.to(args.device)[:, None]
        loss = F.mse_loss(output, target_tmp)
        pred = get_pred_with_levels(output, mid_levels, args)
        if batch_idx % 10 == 0:
            print(f'output: {output.view(-1)}, pred: {pred}, real: {real}')

        correct += pred.eq(real.to(args.device)).sum().item()
        num_samples += len(target)
        loss.backward()
        optimizer.step()
        y_true.append(real.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        if batch_idx % args.train_print_every == 0:
            # print(y_true, y_pred)
            logging.info(
                'Train Epoch: {:04d} | Iter: [{:04d} / {:04d} ({:02d}%)] | Loss: {:.6f}, Acc: {:.6f}, F1 macro: {:.3f}'.format(
                    epoch, batch_idx * len(data), data_loader.sampler.num_samples,
                    int(100. * batch_idx / len(data_loader)), loss.item(), correct / num_samples,
                    f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')))

    acc = correct / num_samples
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    logging.info('Train Epoch: {:04d} | Accuracy: {}/{} ({:.3f}), F1 macro: {:.3f}'.format(
        epoch, correct, num_samples, acc, f1_score(y_true, y_pred, average='macro')))


def test_epoch(epoch, model, data_loader, args, mid_levels):
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target, real in data_loader:
            output = model(data.to(args.device))
            #             test_loss += F.cross_entropy(output, target.to(args.device), reduction='sum').item() # sum up batch loss
            target_tmp = target.to(args.device)[:, None]
            test_loss += F.mse_loss(output, target_tmp).item()
            pred = get_pred_with_levels(output, mid_levels, args)
            correct += pred.eq(real.to(args.device)).sum().item()
            num_samples += len(target)

            y_true.append(real.cpu())
            y_pred.append(pred.cpu())

    test_loss /= num_samples
    acc = correct / num_samples
    logging.info('Test Epoch: {:04d} | Average loss: {:.4f} | Accuracy: {}/{} ({:.3f}) | F1 micro: {:.3f}'.format(
        epoch, test_loss, correct, num_samples, acc, f1_score(y_true, y_pred, average='micro')))
    return acc


def train(args):
    classes = eval(args.classes)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    logging.info('-' * 20 + 'Data' + '-' * 20)
    train_loader, test_loader, num_classes = make_data(args)
    levels = np.linspace(0, 100, len(classes) + 1)
    mid_levels = torch.tensor([(levels[i] + levels[i - 1]) / 2. for i in range(1, len(levels))]).to(args.device)

    logging.info('-' * 20 + 'Model' + '-' * 20)
    model, init_epoch = make_model(args, 1)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logging.info(model)

    logging.info('-' * 20 + 'Train' + '-' * 20)
    model.to(args.device)
    best_acc = 0
    best_epoch = -1
    for epoch in range(init_epoch, init_epoch + args.num_epoch):
        train_epoch(epoch, model, train_loader, optimizer, args, mid_levels)

        if scheduler:
            scheduler.step()

        logging.info('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        if epoch % args.test_print_every == 0:
            current_acc = test_epoch(epoch, model, test_loader, args, mid_levels)
            save_model_epoch(model, args.save_dir, epoch, best_epoch)
            logging.info('Save model at {} epoch'.format(epoch))
            if current_acc > best_acc:
                # save_model_epoch(model, args.save_dir, epoch, best_epoch)
                best_acc, best_epoch = current_acc, epoch
                logging.info('New best acc at {} with {}'.format(epoch, best_acc))

    save_model_epoch(model, args.save_dir, epoch)


if __name__ == '__main__':
    args = parse_args()
    train(args)
