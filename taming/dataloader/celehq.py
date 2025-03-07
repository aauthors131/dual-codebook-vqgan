import os
import dotenv
dotenv.load_dotenv()

# # ------------------------------------------------------------------------------------
# # Enhancing Transformers
# # Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------------------

# import PIL
# from typing import Any, Tuple, Union, Optional, Callable
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import torch
# from torchvision import transforms as T
# from torchvision.datasets import ImageNet
# from torch.utils.data import Dataset
# from torchvision.datasets.folder import ImageFolder
# import os
# import numpy as np
# import pickle

# def is_truncated(filepath):
#     try:
#         PIL.Image.open(filepath)
#     except:
#         return True
#     return False

# def read_images(path):
#     f = open(path)
#     lines = f.readlines()
#     img_id2path = {}
#     for line in lines:
#         id, img_path = line.strip('\n').split(' ')
#         img_id2path[id] = img_path
#     f.close()
#     return img_id2path

# class CelehqTrain(Dataset):
#     def __init__(self, root='celehq', resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75):
#         with open(os.path.join(root,'train','filenames.pickle'), 'rb') as handle:
#             images = pickle.load(handle)
#         self.root = root
#         self.data = []
#         self.segment = []
#         self.n_labels = 19
        
#         for image in images:
#             self.data.append(os.path.join(self.root, 'images', image+'.jpg'))
#             self.segment.append(os.path.join(self.root, 'CelebAMaskHQ-mask', image+'.png'))
#         self.transform = T.Compose([
#             T.Resize(resolution),
#             # T.CenterCrop(resolution),
#             # T.RandomHorizontalFlip(),
#             lambda x: np.asarray(x),
#         ])
#         #print(len(self.labels))
            
#     def __getitem__(self, index):
#         img, seg = self.data[index], self.segment[index]
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = PIL.Image.open(img).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         img = (img/127.5 - 1.0).astype(np.float32)

#         seg = PIL.Image.open(seg)
#         if self.transform is not None:
#             seg = self.transform(seg)
#         seg = seg.astype(np.uint8)[:,:]
#         seg = np.eye(self.n_labels)[seg]

#         return {'image': img, 'segmentation': seg}
#         # return {'image': img, 'class': label}

#     def __len__(self):
#         return len(self.data)

# class CelehqValidation(Dataset):
#     def __init__(self, root='celehq', resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75):
#         with open(os.path.join(root,'test','filenames.pickle'), 'rb') as handle:
#             images = pickle.load(handle)
#         self.root = root
#         self.data = []
#         self.segment = []
#         self.n_labels = 19
        
#         for image in images:
#             self.data.append(os.path.join(self.root, 'images', image+'.jpg'))
#             self.segment.append(os.path.join(self.root, 'CelebAMaskHQ-mask', image+'.png'))
#         self.transform = T.Compose([
#             T.Resize(resolution),
#             # T.CenterCrop(resolution),
#             # T.RandomHorizontalFlip(),
#             lambda x: np.asarray(x),
#         ])
#         #print(len(self.labels))
            
#     def __getitem__(self, index):
#         img, seg = self.data[index], self.segment[index]
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = PIL.Image.open(img).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         img = (img/127.5 - 1.0).astype(np.float32)

#         seg = PIL.Image.open(seg)
#         if self.transform is not None:
#             seg = self.transform(seg)
#         seg = seg.astype(np.uint8)
#         seg = np.eye(self.n_labels)[seg]

#         return {'image': img, 'segmentation': seg}
#         # return {'image': img, 'class': label}

#     def __len__(self):
#         return len(self.data)

# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import PIL
from typing import Any, Tuple, Union, Optional, Callable
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
import os
import numpy as np


def is_truncated(filepath):
    try:
        PIL.Image.open(filepath)
    except:
        return True
    return False

def read_images(path):
    f = open(path)
    lines = f.readlines()
    img_id2path = {}
    for line in lines:
        id, img_path = line.strip('\n').split(' ')
        img_id2path[id] = img_path
    f.close()
    return img_id2path

class CelehqTrain(Dataset):
    def __init__(self, root='/workplace/dataset/cub', resolution: Union[Tuple[int, int], int] = 256):
        # Specify the directory containing .jpeg images
        self.images_path = os.path.join(root, 'train')
        self.data = []

        # List all .jpeg files in the specified directory
        for img_name in os.listdir(self.images_path):
            if img_name.lower().endswith('.jpg'):  # Match case-insensitive .jpeg extension
                self.data.append(os.path.join(self.images_path, img_name))

        # self.transform = T.Compose([
        #     T.CenterCrop(148),
        #     T.Resize((resolution,resolution)),
        #     lambda x: np.asarray(x),  # Convert to numpy array
        # ])
        self.transform = T.Compose([
            T.Resize((resolution,resolution)),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])
            
    def __getitem__(self, index):
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img/127.5 - 1.0).astype(np.float32)
        return {'image': img}
        # return {'image': img, 'class': label}

    def __len__(self):
        return len(self.data)

class CelehqValidation(Dataset):
     def __init__(self, root='/workplace/dataset/cub', resolution: Union[Tuple[int, int], int] = 256):
        # Specify the directory containing .jpeg images
        self.images_path = os.path.join(root, 'val')
        self.data = []

        # List all .jpeg files in the specified directory
        for img_name in os.listdir(self.images_path):
            if img_name.lower().endswith('.jpg'):  # Match case-insensitive .jpeg extension
                self.data.append(os.path.join(self.images_path, img_name))

        # self.transform = T.Compose([
        #     T.CenterCrop(148),
        #     T.Resize((resolution,resolution)),
        #     lambda x: np.asarray(x),  # Convert to numpy array
        # ])

        self.transform = T.Compose([
            T.Resize((resolution,resolution)),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])
            
     def __getitem__(self, index):
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img/127.5 - 1.0).astype(np.float32)
        return {'image': img}
        # return {'image': img, 'class': label}

     def __len__(self):
        return len(self.data)

class CelehqTest(Dataset):
     def __init__(self, root='/workplace/dataset/cub', resolution: Union[Tuple[int, int], int] = 256):
        # Specify the directory containing .jpeg images
        self.images_path = os.path.join(root, 'test')
        self.data = []

        # List all .jpeg files in the specified directory
        for img_name in os.listdir(self.images_path):
            if img_name.lower().endswith('.jpg'):  # Match case-insensitive .jpeg extension
                self.data.append(os.path.join(self.images_path, img_name))

        # self.transform = T.Compose([
        #     T.CenterCrop(148),
        #     T.Resize((resolution,resolution)),
        #     lambda x: np.asarray(x),  # Convert to numpy array
        # ])

         # Use deterministic transformations for testing
        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.CenterCrop(resolution),  # Use CenterCrop instead of RandomCrop for testing
            lambda x: np.asarray(x),  # Convert PIL image to numpy array
        ])
            
     def __getitem__(self, index):
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img/127.5 - 1.0).astype(np.float32)
        return {'image': img}
        # return {'image': img, 'class': label}

     def __len__(self):
        return len(self.data)
