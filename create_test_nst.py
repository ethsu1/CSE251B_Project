from utils import *
from file_utils import *
from model import *
from PIL import Image
from tqdm import tqdm

import copy
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# assigning device based on GPU/CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load config
config_name = 'default'
if len(sys.argv) > 1:
    config_name = sys.argv[1]
config_data = read_file_in_dir('./', config_name + '.json')
if config_data is None:
    raise Exception("Configuration file doesn't exist: ", config_name)

# load config vars
content_dir_path = config_data['test_dataset']['content_dir_path']
content_layers = config_data['experiment']['content_layers']
content_weight = config_data['experiment']['content_weight']
loss_type = config_data['experiment']['loss']
lr = config_data['experiment']['learning_rate']
img_size = config_data['dataset']['img_size']
model_name = config_data['experiment']['model_name']
num_epochs = config_data['experiment']['num_epochs']
optim_type = config_data['experiment']['optimizer']
output_dir_path = config_data['test_dataset']['output_dir_path']
random_noise = config_data['experiment']['random_noise']
style_dir_path = config_data['test_dataset']['style_dir_path']
style_layers = config_data['experiment']['style_layers']
style_weight = config_data['experiment']['style_weight']

for content_file_name in os.listdir(content_dir_path):
    for style_file_name in os.listdir(style_dir_path):
        content_path = os.path.join(content_dir_path, content_file_name)
        style_path = os.path.join(style_dir_path, style_file_name)

        print('\tContent Path: ', content_path)
        print('\tStyle Path: ', style_path)

        content_img = img_loader(content_path, img_size, device)
        style_img = img_loader(style_path, img_size, device)

        assert style_img.size() == content_img.size(
        ), "we need to import style and content images of the same size"

        if model_name == 'resnet34':
            cnn = models.resnet34(pretrained=True).to(device).eval()
        elif model_name == "densenet121":
            cnn = models.densenet121(pretrained=True).to(device).eval()
        else:
            cnn = models.vgg19(pretrained=True).features.to(device).eval()

        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        if random_noise == False:
            input_img = content_img.clone()
        else:
            input_img = (torch.randn(content_img.data.size(), device=device) + 0.5) / 8

        def get_input_optimizer(input_img, optim_type):
            if optim_type == 'adam':
                optimizer = optim.Adam([input_img.requires_grad_()], lr=lr)
            elif optim_type == 'rmsprop':
                optimizer = optim.RMSprop([input_img.requires_grad_()], lr=lr)
            else:
                optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
            return optimizer

        def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps, style_weight, content_weight, content_layers, style_layers):
            if model_name == 'resnet34':
                model, style_losses, content_losses = resnet_model_and_losses(
                    cnn, normalization_mean, normalization_std, style_img, content_img, device, loss_type)
            elif model_name == 'densenet121':
                model, style_losses, content_losses = densenet_model_and_losses(
                    cnn, normalization_mean, normalization_std, style_img, content_img, device, loss_type)
            else:
                model, style_losses, content_losses = get_style_model_and_losses(
                    cnn, normalization_mean, normalization_std, style_img, content_img, device, content_layers, style_layers, loss_type)
            best_img = input_img
            lowest_loss = 1e12
            optimizer = get_input_optimizer(input_img, optim_type)
            run = [0]
            pbar = tqdm(total=1)
            while run[0] <= num_steps:
                def closure():
                    input_img.data.clamp_(0, 1)
                    optimizer.zero_grad()
                    model(input_img)
                    style_score = 0
                    content_score = 0
                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss
                    style_score *= style_weight
                    content_score *= content_weight
                    loss = style_score + content_score
                    loss.backward()
                    run[0] += 1
                    pbar.update(1)
                    return style_score + content_score
                loss = optimizer.step(closure).item()
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_img = input_img.clone()
            pbar.close()
            best_img.data.clamp_(0, 1)
            return best_img

        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, num_epochs, style_weight, content_weight, content_layers, style_layers)

        plt.figure()
        plt.ioff()
        image = output.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        plt.imshow(image)
        path = output_dir_path + model_name + '_' + loss_type + '_' + optim_type + '/'
        if os.path.isdir(path) == False:
            os.mkdir(path)
        file_output = path + os.path.splitext(os.path.basename(content_path))[
            0] + '_' + os.path.splitext(os.path.basename(style_path))[0] + '.png'
        plt.axis('off')
        plt.savefig(file_output, bbox_inches='tight', pad_inches=0)

print('Done! Check the specified directory for your style-transferred images.')