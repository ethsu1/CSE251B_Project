from utils import *
from file_utils import *
from model import *
from PIL import Image

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
content_layers = config_data['experiment']['content_layers']
content_path = config_data['dataset']['content_image_path']
content_weight = config_data['experiment']['content_weight']
lr = config_data['experiment']['learning_rate']
img_size = config_data['dataset']['img_size']
model_name = config_data['experiment']['model_name']
num_epochs = config_data['experiment']['num_epochs']
output_title = config_data['experiment']['output_title']
random_noise = config_data['experiment']['random_noise']
style_path = config_data['dataset']['style_image_path']
style_weight = config_data['experiment']['style_weight']
style_layers = config_data['experiment']['style_layers']
loss_type = config_data['experiment']['loss']
optim_type = config_data['experiment']['optimizer']

# get image
content_img = img_loader(content_path, img_size, device)
style_img = img_loader(style_path, img_size, device)

assert style_img.size() == content_img.size(
), "we need to import style and content images of the same size"

# turn on interactive mode
plt.ion()

# show style and content image
plt.figure()
imshow(style_img, title='Style Image')
plt.figure()
imshow(content_img, title='Content Image')

# load frozen pretrained model (depending on config)
if model_name == 'resnet34':
  cnn = models.resnet34(pretrained=True).to(device).eval()
elif model_name == "densenet121":
  cnn = models.densenet121(pretrained=True).to(device).eval()
else:
  # default
  cnn = models.vgg19(pretrained=True).features.to(device).eval()

# used to build style transfer model
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# copy of content image
if random_noise == False:
  input_img = content_img.clone()
# random noise
else:
  input_img = (torch.randn(content_img.data.size(), device=device) + 0.5) / 8  # shifted to avoid clamping


def get_input_optimizer(input_img, optim_type):
  if optim_type == 'adam':
    optimizer = optim.Adam([input_img.requires_grad_()], lr=lr)
  elif optim_type == 'rmsprop':
    optimizer = optim.RMSprop([input_img.requires_grad_()], lr=lr)
  else:
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
  return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps, style_weight, content_weight, content_layers, style_layers):
  print('Building the style transfer model..')
  if model_name == 'resnet34':
    print('Building ResNet34 model!')
    model, style_losses, content_losses = resnet_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img, device, loss_type)
  elif model_name == 'densenet121':
    print('Building DenseNet121 model!')
    model, style_losses, content_losses = densenet_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img, device, loss_type)
  else:
    print('Building VGG19 model!')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img, device, content_layers, style_layers, loss_type)
  best_img = input_img
  lowest_loss = 1e12
  optimizer = get_input_optimizer(input_img, optim_type)
  print('Optimizing..')
  style_scores = []
  content_scores = []
  total_scores = []
  run = [0]
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
      if run[0] % 50 == 0:
        print("run {}:".format(run))
        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            style_score.item(), content_score.item()))
        print()
      style_scores.append(style_score.item())
      content_scores.append(content_score.item())
      total_scores.append((style_score + content_score).item())
      return style_score + content_score
    loss = optimizer.step(closure).item()
    if loss < lowest_loss:
      lowest_loss = loss
      best_img = input_img.clone()
  best_img.data.clamp_(0, 1)
  return best_img, lowest_loss, style_scores, content_scores, total_scores


output, lowest_loss, style_scores, content_scores, total_scores = run_style_transfer(
    cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, num_epochs, style_weight, content_weight, content_layers, style_layers)

# print lowest total loss
print('LOWEST TOTAL LOSS: ', lowest_loss)

# print style-transferred image
plt.figure()
# no title for easier formatting
imshow(output)
plt.ioff()
path = './images/output/' + model_name + '/'
if os.path.isdir(path) == False:
    os.mkdir(path)
file_output = path + os.path.splitext(os.path.basename(content_path))[
    0] + '+' + os.path.splitext(os.path.basename(style_path))[0] + output_title + '.png'
# removing axis
plt.axis('off')
# saving figure
plt.savefig(file_output, bbox_inches='tight', pad_inches=0)
plt.show()

title = config_data['loss_plot']['title']
y_title = 'Loss'
x_title = 'Number of steps'
y_data = np.matrix([style_scores, content_scores, total_scores])
legend = ['Style Loss', 'Content Loss', 'Total Loss']

if config_data['loss_plot']['plot_graph'] == True:
  plot_stats(title, y_title, x_title, y_data, legend=legend)
