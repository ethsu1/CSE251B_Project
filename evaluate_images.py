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

parser = argparse.ArgumentParser(description='Classifying Images as Anime/Real w/ Trained Model')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default='0', help='index of gpus to use (for two, use --gpu 0,1)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--image-dir', default='', type=str, metavar='PATH',
                    help='path to directory for images to evaluate (e.g. ./eval_images/ or /.test_data/resnet34_l2_lbfgs/)')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH',
                    help='path for results (e.g. ./results/)')
parser.add_argument('--img-size', default=256, type=int,
                    metavar='N', help='dimension to resize images to (square, default: 256)')
parser.add_argument('--model-path', default='', type=str, metavar='PATH',
                    help='path to pretrained model (e.g. models/model_best.pth.tar')
parser.add_argument('--no-color', dest='no_color', action='store_true',
                    help='make all evaluation images monochrome')

def main():
    global args
    args = parser.parse_args()
    out_dir = args.save_dir

    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)

    print('\n\n\n\n\n\n\n')
    print('Using {} Workers to Load Data\nBatch Size: {}\n'.format(args.workers, args.batch_size))

    assert (args.image_dir != ''), 'Must specify evaluation image directory'

    # Check if model exists
    assert os.path.isfile(args.model_path), 'Selected model does not exist'

    checkpoint = torch.load(args.model_path)
    fcn_model = torchvision.models.resnet18(progress=True)
    fcn_model.fc = nn.Linear(512, 2)
    fcn_model.load_state_dict(checkpoint['state_dict'])

    # Select GPUs
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # Load Dataset
    image_dataset = EvalDataset(img_size=args.img_size, image_dir=args.image_dir, grayscale=args.no_color)

    image_loader = DataLoader(dataset=image_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # Data parallelism w multiple GPUs
    if len(args.gpu) > 1:
        fcn_model = torch.nn.DataParallel(fcn_model, device_ids=range(len(args.gpu))).cuda()
    else:
        fcn_model = fcn_model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    val_loss, val_acc, predictions = val(fcn_model, image_loader, criterion)

    np.save(out_dir + 'eval_loss.npy', val_loss)
    np.save(out_dir + 'eval_acc.npy', val_acc)
    np.save(out_dir + 'predictions.npy', predictions)

def val(model, val_loader, criterion):

    num_images = len(val_loader.dataset)
    # Switch to Evaluation Mode
    model.eval()

    # Define Variables and Load onto GPU
    samples = torch.zeros(1)
    loss = torch.zeros(1).cuda()
    acc_val = torch.zeros(1).cuda()
    predictions = torch.zeros(num_images).cuda()

    ts = time.time()
    with torch.no_grad():
        for iteration, (X, Y) in enumerate(val_loader):
            inputs = X.cuda()
            labels = Y.cuda()

            cur_batch = float(Y.shape[0])

            output = model(inputs)
            preds = predict(output)
            predictions[samples:samples+cur_batch] = preds
            loss += criterion(output, labels).detach()
            acc_val += get_acc(preds, labels)

            samples += cur_batch

    samples = samples / args.batch_size
    print('\nTime Elapsed: {:.4f}\nVal Loss: {:.4f}\nVal Acc: {:.4f}\n'
          .format(time.time() - ts, loss.item()/samples.item(), acc_val.item()/samples.item()))
    return (loss.item()/samples.item()), (acc_val.item()/samples.item()), predictions.cpu().numpy()


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


