import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
  def __init__(self, target,):
    super(ContentLoss, self).__init__()
    self.target = target.detach()

  def forward(self, input):
    self.loss = F.mse_loss(input, self.target)
    return input


def gram_matrix(input):
  a, b, c, d = input.size()
  features = input.view(a * b, c * d)
  G = torch.mm(features, features.t())
  return G.div(a * b * c * d)


class StyleLoss(nn.Module):
  def __init__(self, target_feature):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_feature).detach()

  def forward(self, input):
    G = gram_matrix(input)
    self.loss = F.mse_loss(G, self.target)
    return input


class Normalization(nn.Module):
  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    self.mean = torch.tensor(mean).view(-1, 1, 1)
    self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
    return (img - self.mean) / self.std


# layers used by Gatys et al.
'''content_layers_default = ['conv4_2']
style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']'''


def resnet_model_and_losses(cnn, style_img, content_img, device):
  cnn = copy.deepcopy(cnn)
  content_losses = []
  style_losses = []
  i = 1

  # refer to resnet_architecture.txt for ResNet34 architecture
  # Benson: this is all hard-coded for now since I was just testing stuff out
  # lr = 1.5 is pretty good

  model = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1)

  # layer 1 - style
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[4].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 2
  model.add_module('layer2', cnn.layer2)

  # layer 2 - style
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[5].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 3
  model.add_module('layer3', cnn.layer3)

  # layer 3 - style
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[6].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 4 (in between 0 and 1)
  model.add_module('layer4', nn.Sequential(cnn.layer4[0]))

  # layer 4 - style
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[7].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 4 - content
  target = model(content_img).detach()
  content_loss = ContentLoss(target)
  model[7].add_module("content_loss", content_loss)
  content_losses.append(content_loss)

  return model, style_losses, content_losses


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, device, content_layers, style_layers):
  # RENAMING LAYERS
  cnn = copy.deepcopy(cnn)
  normalization = Normalization(
      normalization_mean, normalization_std).to(device)
  content_losses = []
  style_losses = []
  model = nn.Sequential(normalization)
  block, i = 1, 1
  for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
      name = 'conv{}_{}'.format(block, i)
    elif isinstance(layer, nn.ReLU):
      name = 'relu{}_{}'.format(block, i)
      layer = nn.ReLU(inplace=False)
      i += 1
    elif isinstance(layer, nn.MaxPool2d):
      name = 'pool_{}'.format(block)
      # replace with average pooling as suggested by Gatys et al.
      layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
      block += 1
      i = 1
    elif isinstance(layer, nn.BatchNorm2d):
      name = 'bn{}_{}'.format(block, i)
    else:
      raise RuntimeError(
          'Unrecognized layer: {}'.format(layer.__class__.__name__))
    model.add_module(name, layer)
    # GETTING LOSSES
    if name in content_layers:
      target = model(content_img).detach()
      content_loss = ContentLoss(target)
      model.add_module("content_loss{}_{}".format(block, i), content_loss)
      content_losses.append(content_loss)
    if name in style_layers:
      target_feature = model(style_img).detach()
      style_loss = StyleLoss(target_feature)
      model.add_module("style_loss{}_{}".format(block, i), style_loss)
      style_losses.append(style_loss)
  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
      break
  model = model[:(i + 1)]
  return model, style_losses, content_losses
