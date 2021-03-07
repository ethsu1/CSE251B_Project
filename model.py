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


def resnet_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, device):
  # ideal: lr = 0.75, style_weight = 1000000000
  # refer to resnet_architecture.txt for ResNet34 architecture
  cnn = copy.deepcopy(cnn)
  content_losses = []
  style_losses = []
  i = 1

  normalization = Normalization(
      normalization_mean, normalization_std).to(device)

  model = nn.Sequential(normalization, cnn.conv1,
                        cnn.bn1, cnn.relu, cnn.maxpool)

  # layer 1 (0)
  model.add_module('layer1', nn.Sequential(cnn.layer1[0]))

  # layer 1 - style (in between 0 and 1)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[5].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 1 (1 - 2)
  model[5].add_module('layer1_1', cnn.layer1[1])
  model[5].add_module('layer1_2', cnn.layer1[2])

  # layer 2 (0)
  model.add_module('layer2', nn.Sequential(cnn.layer2[0]))

  # layer 2 - style (in between 0 and 1)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[6].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 2 (1 - 2)
  model[6].add_module('layer2_1', cnn.layer2[1])
  model[6].add_module('layer2_2', cnn.layer2[2])
  model[6].add_module('layer2_3', cnn.layer2[3])

  # layer 3 (0)
  model.add_module('layer3', nn.Sequential(cnn.layer3[0]))

  # layer 3 - content (in between 0 and 1)
  target = model(content_img).detach()
  content_loss = ContentLoss(target)
  model[7].add_module("content_loss", content_loss)
  content_losses.append(content_loss)

  # layer 3 - style (in between 0 and 1)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[7].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 3 (1 - 3)
  model[7].add_module('layer3_1', cnn.layer3[1])
  model[7].add_module('layer3_2', cnn.layer3[2])
  model[7].add_module('layer3_3', cnn.layer3[3])

  # layer 3 - style (between 3 and 4)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[7].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  # layer 3 (4 - 5)
  model[7].add_module('layer3_4', cnn.layer3[4])
  model[7].add_module('layer3_5', cnn.layer3[5])

  # layer 4 (0)
  model.add_module('layer4', nn.Sequential(cnn.layer4[0]))

  # layer 4 - style (in between 0 and 1)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[8].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  i += 1

  return model, style_losses, content_losses

def densenet_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, device):
  # refer to desnenet_architecture.txt for DenseNet121 architecture
  #taking style losses for the convolutional layers between denseblocks 
  cnn = copy.deepcopy(cnn)
  content_losses = []
  style_losses = []
  i = 1

  normalization = Normalization(
      normalization_mean, normalization_std).to(device)

  model = nn.Sequential(normalization, cnn.features.conv0, cnn.features.norm0, cnn.features.relu0, cnn.features.pool0)
  
  # denseblock 1
  model.add_module('denseblock1', cnn.features.denseblock1)
 
  model.add_module('transition1',  nn.Sequential())
  model[6].add_module('norm',  cnn.features.transition1.relu)
  model[6].add_module('relu',  torch.nn.ReLU(inplace=False))
  # layer 1 - style (in between denseblock1 and denseblock2)
  model[6].add_module('conv', cnn.features.transition1.conv)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[6].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  target = model(content_img).detach()
  content_loss = ContentLoss(target)
  model[6].add_module("content_loss_{}".format(i), content_loss)
  content_losses.append(content_loss)
  i += 1
  model[6].add_module('pool',  cnn.features.transition1.pool)

  #denseblock2
  model.add_module('denseblock2', cnn.features.denseblock2)

  model.add_module('transition2', nn.Sequential())
  model[8].add_module('norm',  nn.Sequential(cnn.features.transition2.norm))
  model[8].add_module('relu',  torch.nn.ReLU(inplace=False))

  # layer 2 - style (in between denseblock2 and denseblock3)
  model[8].add_module('conv', cnn.features.transition2.conv)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[8].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  target = model(content_img).detach()
  content_loss = ContentLoss(target)
  model[8].add_module("content_loss_{}".format(i), content_loss)
  content_losses.append(content_loss)
  i += 1
  model[8].add_module('pool',  cnn.features.transition2.pool)
  
  #denseblock3
  model.add_module('denseblock3', cnn.features.denseblock3)


  model.add_module('transition3', nn.Sequential())
  model[10].add_module('norm',  cnn.features.transition3.norm)
  model[10].add_module('relu', torch.nn.ReLU(inplace=False))
   
  #layer 3 - style (in between denseblock3 and denseblock4)
  model[10].add_module('conv', cnn.features.transition3.conv)
  target_feature = model(style_img).detach()
  style_loss = StyleLoss(target_feature)
  model[10].add_module("style_loss_{}".format(i), style_loss)
  style_losses.append(style_loss)
  target = model(content_img).detach()
  content_loss = ContentLoss(target)
  model[10].add_module("content_loss_{}".format(i), content_loss)
  content_losses.append(content_loss)
  i += 1
  return model, style_losses, content_losses

# layers used by Gatys et al.
'''content_layers_default = ['conv4_2']
style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']'''


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, device, content_layers, style_layers):
  # for VGG19: lr = 0.99, style_weight = 1000000
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