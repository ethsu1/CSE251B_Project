from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import torch

ANIME_INDEX_FN = 'anime_data/anime_data_index.txt'
REAL_INDEX_FN = 'real_data/real_data_index.txt'
IMAGE_DEFAULT_SIZE = 256

class ImageDataset(Dataset):
    def __init__(self, mode='train', img_size=IMAGE_DEFAULT_SIZE, anime_index=ANIME_INDEX_FN, real_index=REAL_INDEX_FN):
        """

        Args:
        """
        self.data, self.idx = self._read_index_files(mode, anime_index, real_index)
        self.img_size = img_size


        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.resize = transforms.Compose([transforms.Resize(img_size, interpolation=2),
                                          transforms.CenterCrop(img_size)])


    def _read_index_files(self, data_mode, anime_index, real_index):
        with open(anime_index, 'r') as f:
            anime_fn = f.read().split('\n')[:-1]
        with open(real_index, 'r') as f:
            real_fn = f.read().split('\n')[:-1]
        if data_mode == 'train':
            print('Using {} Anime Images.\nUsing {} Real Images.'.format(len(anime_fn), len(real_fn)))
        if data_mode == 'train':
            anime_fn = anime_fn[0:int(len(anime_fn)*0.9)]
            real_fn = real_fn[0:int(len(real_fn)*0.9)]
        else:
            anime_fn = anime_fn[int(len(anime_fn)*0.9):-1]
            real_fn = real_fn[int(len(real_fn)*0.9):-1]
        anime_lbl = [0] * len(anime_fn)
        real_lbl = [1] * len(real_fn)
        data = anime_fn + real_fn
        lbl = anime_lbl + real_lbl
        return data, lbl

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        fn = self.data[item]
        label = self.idx[item]

        image = Image.open(fn).convert('RGB')
        image = self.resize(image)
        image = self.normalize(image)

        return image, label

