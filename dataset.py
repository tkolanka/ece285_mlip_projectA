'''
This file contains the class for creation of the image captioning MS COCO 2014 dataset. This is used for creating train and validation sets.
It will normally be used by a dataloader for batch training and validation.
'''

import json, os
import torch
from PIL import Image
import torch.utils.data as td
import torchvision as tv

class COCO14Dataset(td.Dataset):
    '''
    A class of the COCO 2014 dataset that will be used for image captioning.
    This class mainly defines the process for getting an image and its corresponding captions from the training or validation set.
    Make sure that the images are stored inside an imgs folder in the directory pointed to by datapath. Additionally, the 
    training and validation images must be in separate folders namely train2014 and val2014.
    
    Arguments:
        transform (torchvision.transforms.transforms.Compose): Set of transforms to be applied to the images
                                                               Default = tv.transforms.Compose([
                                                                         tv.transforms.Resize((224, 224)),
                                                                         tv.transforms.ToTensor(),
                                                                         tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                 std=[0.229, 0.224, 0.225])
                                                                         ])
        data_path (str): Complete path of the main directory containing the json files and the images. Default = 'data/coco'
        mode (str): Dataset to be accessed - 'train' or 'val'. Default = 'train'
        image_size (tuple): Final image size that the image must be resized to. Default = (224, 224)
    '''
    
    def __init__(self, transform, data_path, mode='train', image_size=(224,224)):
        super(COCO14Dataset, self).__init__()
        self.mode = mode
        if transform is not None:
            self.transform = transform
        else:
            self.transform = tv.transforms.Compose([
                             tv.transforms.Resize(image_size),
                             tv.transforms.ToTensor(),
                             tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])
        self.image_size = image_size
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(mode), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(mode), 'r'))

    def __getitem__(self, index):
        # Access the image at the corresponding location and transform it as specified
        img_path = self.img_paths[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        # For training, return the image and its corresponding captions
        if self.mode == 'train':
            return torch.FloatTensor(img), torch.tensor(self.captions[index])

        # For validation, return the image and all the captions associated with that image in the complete dataset
        matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        return torch.FloatTensor(img), torch.tensor(self.captions[index]), torch.tensor(all_captions)

    def __len__(self):
        return len(self.captions)
