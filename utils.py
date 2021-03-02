from PIL import Image

import matplotlib.pyplot as plt
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
