from PIL import Image

import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms

unloader = transforms.ToPILImage()


def img_loader(image_name, img_size, device):
  loader = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.CenterCrop((img_size, img_size)),
      transforms.ToTensor()])
  image = Image.open(image_name)
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)


def imshow(tensor, title=None):
  image = tensor.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  plt.imshow(image)
  if title is not None:
    plt.title(title)
  plt.pause(0.001)


def plot_stats(title, y_title, x_title, y_data, x_data=[], legend=[], path="./images/figures/"):
  '''
  plot graph based on arguments
  Args:
      title - title of graph
      y_title - y-axis title
      x_title - x-axis title
      y_data - data along y axis
      x_data - data along x axis
      legend - list of strings corresponding to each set of y_data 
          (i.e., if you want to plot multiple sets of data on one graph)
  '''
  if os.path.isdir(path) == False:
    os.mkdir(path)
  plt.figure()
  length = y_data.shape[1]
  if len(x_data) == 0:
    x = [i for i in range(1, length + 1)]
  else:
    x = x_data
  for i, y in enumerate(y_data):
    plt.plot(x, y.T, label=legend[i])
  plt.ylabel(y_title)
  plt.xlabel(x_title)
  plt.title(title)
  plt.legend()
  title = title.replace(" ", "_")
  plt.savefig(path + title + ".png")
  plt.show()
