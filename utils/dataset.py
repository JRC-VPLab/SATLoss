import os
import re

import cv2 as cv
import torch

def natural_key(str1):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', str1)]

class Topo_dataloader(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        if mode == 'train':
            self.img_root = os.path.join(root, 'train')
            self.gt_root = os.path.join(root, 'train_labels')
        elif mode == 'val':
            self.img_root = os.path.join(root, 'val')
            self.gt_root = os.path.join(root, 'val_labels')
        elif mode == 'test':
            self.img_root = os.path.join(root, 'test')
            self.gt_root = os.path.join(root, 'test_labels')

        self.img_list = [f for f in sorted(os.listdir(self.img_root), key=natural_key) if not f.startswith('.')]
        self.gt_list = [f for f in sorted(os.listdir(self.gt_root), key=natural_key) if not f.startswith('.')]

        assert len(self.img_list) == len(self.gt_list)
        print('{} {} images'.format(len(self.img_list), mode))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img = cv.imread(os.path.join(self.img_root, self.img_list[item]))
        gt = cv.imread(os.path.join(self.gt_root, self.gt_list[item]))

        return img / 255, gt / 255, self.img_list[item]
