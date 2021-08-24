import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import cfg
torch.manual_seed(17)

mark = '\\'

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomRotation(30)
    ]
)

class DenDataset(Dataset):
    def __init__(self, file_name):
        super(DenDataset, self).__init__()
        f = open(file_name, "r")
        self.img_paths = f.read().splitlines()

    def __len__(self):
        return len(self.img_paths) - 6

    def __getitem__(self, item):
        imgs_a = []
        imgs_b = []

        for i in range(cfg.WINDOW_LIMITS):

            img_a, img_b = self.img_paths[item + i].split('\t')
            img_a, img_b = cv2.imread(img_a), cv2.imread(img_b)
            img_a, img_b = cv2.resize(img_a, (cfg.IMG_SIZE, cfg.IMG_SIZE)), cv2.resize(img_b, (cfg.IMG_SIZE, cfg.IMG_SIZE))
            img_a, img_b = trans(img_a), trans(img_b)
            imgs_a.append(img_a)
            imgs_b.append(img_b)

        imgs_a = torch.stack(imgs_a, dim=0)
        imgs_b = torch.stack(imgs_b, dim=0)
        imgs = torch.stack([imgs_a, imgs_b], dim=0)

        one_hot = self.img_paths[item + cfg.WINDOW_LIMITS // 2].split(mark)[-2][:6]
        one_hot = [int(i) for i in one_hot]
        label = torch.tensor([one_hot.index(1)])


        return imgs, label


train_data = DenDataset(r'C:\Users\liewei\Desktop\DenseData\traindata.txt')
evens = list(range(0, 1265, 2))
overfit_test = Subset(train_data, evens)
train_loader = DataLoader(train_data, batch_size=None, shuffle=True)


valid_data = DenDataset(r'C:\Users\liewei\Desktop\validation\valdata.txt')
valid_loader = DataLoader(valid_data, batch_size=None, shuffle=True)

if __name__ == "__main__":
    for i in range(1265):
        print(train_data[i])
