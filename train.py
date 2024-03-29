"""
This file is used to train the model

You can change the configs like view1, view2, view1_status, view2_status, model_name, num_classes, epoch_num, in the
config.py

The model_name list is: DualStreamC3D, SlowFast_Multiview
The dataset list is: RHM

it saves the best model and the results in the output folder.


Author: Mohammad Hossein Bamorovat Abadi
Email: m.bamorovvat@gmail.com
License: GNU General Public License (GPL) v3.0
"""


import os
import time
import numpy as np
import torch
import glob
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataloader import VideoDataset
from models import C3D_Multiview, SlowFast_Multiview
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

Debug = False
Save_path = '/home/abbas/RHM_full/output'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if Debug:
    print("Device being used:", device)

# identify the prediction and truth
y_pred = []
y_true = []
best_test_acc1 = 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, (inputs1, inputs2, labels) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = labels.to(device)
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss

        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs1.size(0))
        top1.update(prec1.item(), inputs1.size(0))
        top5.update(prec5.item(), inputs1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

    print('---------------------- Train ---------------------------------')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
    print(print_string)
    print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
        data_time=data_time.val, batch_time=batch_time.val)
    print(print_string)
    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
        top1_acc=top1.avg, top5_acc=top5.avg)
    print(print_string)

    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)


def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (inputs1, inputs2, labels) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs1.size(0))
            top1.update(prec1.item(), inputs1.size(0))
            top5.update(prec5.item(), inputs1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    print('---- Validation ----')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
    print(print_string)
    print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
        data_time=data_time.val, batch_time=batch_time.val)
    print(print_string)
    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
        top1_acc=top1.avg, top5_acc=top5.avg)
    print(print_string)

    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)

    return top1.avg, top5.avg


def test(model, test_dataloader, epoch, criterion, optimizer, writer, better):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (inputs1, inputs2, labels) in enumerate(test_dataloader):
            data_time.update(time.time() - end)
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)

            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction
            labels_1 = labels.data.cpu().numpy()
            y_true.extend(labels_1)  # Save Truth

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs1.size(0))
            top1.update(prec1.item(), inputs1.size(0))
            top5.update(prec5.item(), inputs1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    print('---- Test ----')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(test_dataloader))
    print(print_string)
    print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
        data_time=data_time.val, batch_time=batch_time.val)
    print(print_string)
    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
        top1_acc=top1.avg, top5_acc=top5.avg)
    print(print_string)

    writer.add_scalar('test_loss_epoch', losses.avg, epoch)
    writer.add_scalar('test_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('test_top5_acc_epoch', top5.avg, epoch)

    top1_acc = top1.avg
    global best_test_acc1
    if top1_acc > best_test_acc1:

        # constant for classes
        classes = ['Bending', 'SittingDown', 'ClosingCan', 'Reaching', 'Walking', 'Drinking', 'StairsClimbingUp',
                   'StairsClimbingDown', 'StandingUp', 'OpeningCan', 'CarryingObject', 'Cleaning', 'PuttingDownObjects',
                   'LiftingObject']

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(15, 10))
        sn.heatmap(df_cm, annot=True)

        save_name = (Save_path + 'confM' + '-' + params['model_name'] + '-' + params['view1'] + '-' +
                     params['view2'] + params['view1_status'] + '-' + params['view2_status'] + '.' + 'png')
        plt.savefig(save_name)

    best_test_acc1 = max(top1_acc, best_test_acc1)


def main():

    if Debug:
        print('Start training...')

    val_top1, val_top5 = 0.0, 0.0
    cudnn.benchmark = False
    # useTest = params['useTest']

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    save_dir_models = os.path.join(save_dir, 'models')
    saveName = (params['model_name'] + '-' + params['view1'] + '-' + params['view2'] + '-' + params['view1_status']
                + '-' + params['view2_status'])

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join('log', cur_time)

    if Debug:
        print('save_dir= ', save_dir)
        print('save_dir_models= ', save_dir_models)
        print('saveName= ', saveName)
        print('logdir= ', logdir)
        print('cur_time= ', cur_time)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_dir_models):
        os.makedirs(save_dir_models)

    writer = SummaryWriter(log_dir=logdir)

    if Debug:
        print("Loading dataset ...")
        print('loading train_dataloader ...')

    train_dataloader = \
        DataLoader(
            VideoDataset(view1=params['view1'], view2=params['view2'], view1_status=params['view1_status'],
                         view2_status=params['view2_status'], split='train', clip_len=params['clip_len']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    if Debug:
        print('loading val_dataloader ...')

    val_dataloader = \
        DataLoader(
            VideoDataset(view1=params['view1'], view2=params['view2'], view1_status=params['view1_status'],
                         view2_status=params['view2_status'], split='val', clip_len=params['clip_len']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    if Debug:
        print('loading test_dataloader ...')

    test_dataloader = \
        DataLoader(
            VideoDataset(view1=params['view1'], view2=params['view2'], view1_status=params['view1_status'],
                         view2_status=params['view2_status'], split='test', clip_len=params['clip_len']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    if Debug:
        print("loading model ... ")

    modelName = params['model_name']

    if Debug:
        print('modelName= ', modelName)

    if modelName == 'DualStreamC3D':
        model = C3D_Multiview.DualStreamC3D(num_classes=params['num_classes'], pretrained=params['pretrained'])
        train_params = [{'params': C3D_Multiview.get_1x_lr_params(model), 'lr': params['learning_rate']},
                        {'params': C3D_Multiview.get_10x_lr_params(model), 'lr': params['learning_rate'] * 10}]

    elif modelName == 'SlowFast_Multiview':
        model = SlowFast_Multiview.resnet50(class_num=params['num_classes'])
        train_params = model.parameters()

    else:
        print('The model can not be fined')
        raise NotImplementedError

    model = model.to(device)
    model = nn.DataParallel(model, device_ids=params['gpu'])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(train_params, lr=params['learning_rate'], momentum=params['momentum'],
                          weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    best_acc1 = 0.0
    better = False
    for epoch in range(params['epoch_num']):
        for phase in ['train', 'val']:
            if phase == 'train':
                train(model, train_dataloader, epoch, criterion, optimizer, writer)
            elif phase == 'val':
                val_top1, val_top5 = validation(model, val_dataloader, epoch, criterion, optimizer, writer)
            scheduler.step()

            acc1 = val_top1
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'best_acc1': best_acc1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(), }, os.path.join(save_dir_models, saveName + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir_models, saveName + '.pth')))
            better = True
            print("Saved Epoch Number is: ", epoch)

        test(model, test_dataloader, epoch, criterion, optimizer, writer, better)

    writer.close()


if __name__ == '__main__':
    main()
