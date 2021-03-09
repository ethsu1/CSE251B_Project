import argparse
import os
import shutil
import sys
from PIL import Image
from torchvision import utils
from dataloader import *
import torch
import torch.nn as nn

import torchvision

import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import numpy as np

parser = argparse.ArgumentParser(description='Training Anime/Real Classifier')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default='0', help='index of gpus to use (for two, use --gpu 0,1)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run (default: 5)')
parser.add_argument('--arch', metavar='ARCH', default='basic_fcn', type=str,
                    help='model architecture: ' + ' (default: basic_fcn)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--save-dir', default='extra/', type=str, metavar='PATH',
                    help='path to directory for saved outputs (default: extra/)')
parser.add_argument('--img-size', default=256, type=int,
                    metavar='N', help='dimension to resize images to (square, default: 256)')



def main():
    global args
    args = parser.parse_args()
    out_dir = args.save_dir

    print('\n\n\n\n\n\n\n')
    print('Using {} Workers to Load Data\nBatch Size: {}\nLearning Rate: {}\n'
          'Saving Output to {}'.format(args.workers, args.batch_size, args.lr, args.save_dir))

    # Check if Output Directory Exists
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Select GPUs
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # Load Datasets
    train_dataset = ImageDataset(mode='train', img_size=args.img_size)
    val_dataset = ImageDataset(mode='val', img_size=args.img_size)


    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)


    fcn_model = torchvision.models.resnet18(pretrained=True, progress=True)
    for param in fcn_model.parameters():
        param.requires_grad = False
    fcn_model.fc = nn.Linear(512, 2)


    # Data parallelism w multiple GPUs
    if len(args.gpu) > 1:
        fcn_model = torch.nn.DataParallel(fcn_model, device_ids=range(len(args.gpu))).cuda()
    else:
        fcn_model = fcn_model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, fcn_model.parameters()), lr=args.lr)
    val_loss = np.zeros(args.epochs)
    val_acc = np.zeros(args.epochs)
    train_loss = np.zeros(args.epochs)

    best_loss = 1000000.
    start_time = time.time()
    for n in range(args.epochs):
        train_loss[n] = train(train_loader, fcn_model, criterion, optimizer, n+1)
        val_loss[n], val_acc[n] = val(n+1, fcn_model, val_loader, criterion)

        is_best = val_loss[n] < best_loss
        if is_best:
            best_loss = val_loss[n]
        save_checkpoint({
            'epoch': n,
            'state_dict': fcn_model.state_dict(),
            'best_val_loss': val_loss[n],
            'optimizer': optimizer.state_dict(),
        }, is_best, save_directory=out_dir)
        np.save(out_dir + 'val_loss.npy', val_loss)
        np.save(out_dir + 'val_acc.npy', val_acc)
        np.save(out_dir + 'train_loss.npy', train_loss)
    print('Total Training Time: {} Hours'.format((time.time() - start_time)/3600))



def train(train_loader, model, criterion, optimizer, epoch):

    print('Starting Training, Epoch {}'.format(epoch))

    # Switch to train mode
    model.train()

    # Define cost variable and load onto GPU
    cost = torch.zeros(1).cuda()

    ts = time.time()
    for iteration, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = X.cuda()
        labels = Y.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        cost += loss.detach()
        loss.backward()
        optimizer.step()
        if iteration % 50 == 0:
            print('Batch {} finished. Time Elapsed Since Epoch Started: {:.4f}'.format(iteration, time.time() - ts))
            print('Batch Loss: {:.4f}. Batch Accuracy: {:.4f}.'.format(loss.item(), get_acc(predict(outputs), labels)))


    print("\nFinished Epoch {}\nTime Elapsed (Total): {:.4f}\nTraining Loss: {:.4f}\n".format(epoch, time.time() - ts, cost.item()/len(train_loader)))
    return (cost.item())/len(train_loader)


def val(epoch, model, val_loader, criterion):

    print('Evaluating on Validation Data, Epoch {}'.format(epoch))

    # Switch to Evaluation Mode
    model.eval()

    # Define Variables and Load onto GPU
    samples = torch.zeros(1)
    loss = torch.zeros(1).cuda()
    acc_val = torch.zeros(1).cuda()

    ts = time.time()
    with torch.no_grad():
        for iteration, (X, Y) in enumerate(val_loader):

            inputs = X.cuda()
            labels = Y.cuda()

            cur_batch = float(Y.shape[0])
            samples += cur_batch

            output = model(inputs)
            preds = predict(output)
            loss += criterion(output, labels).detach()


            acc_val += get_acc(preds, labels)

    samples = samples / args.batch_size
    print('\nTime Elapsed: {:.4f}\nVal Loss: {:.4f}\nVal Acc: {:.4f}\n'
          .format(time.time() - ts, loss.item()/samples.item(), acc_val.item()/samples.item()))
    return (loss.item()/samples.item()), (acc_val.item()/samples.item())


def predict(network_out):
    return torch.argmax(network_out, dim=1)

def get_acc(pred, lbl):

    return torch.sum(pred == lbl) / lbl.shape[0]


def save_checkpoint(state, is_best, save_directory=''):
    cur_fn = os.path.join(save_directory, 'checkpoint.pth.tar')
    best_fn = os.path.join(save_directory, 'model_best.pth.tar')
    torch.save(state, cur_fn)
    if is_best:
        shutil.copyfile(cur_fn, best_fn)


if __name__ == "__main__":
    main()


