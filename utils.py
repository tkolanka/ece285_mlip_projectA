'''
This file contains some generic functions/classes that are needed in the overall implementation. It contains definitions of the accuracy function needed during training and validation. It also has a function to compute the actual length of a caption.
'''

import torch
import torchvision as tv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Transform to be applied on the image prior to processing - Used for the data visualization step
data_transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class AverageMeter(object):
    '''
    Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    This class can be used to maintain average statistics over multiple iterations of training and validation.
    '''
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    '''
    Function to compute the accuracy of prediction.
    
    Arguments:
        preds: Predictions of the network
        targets: Expected output of the network
        k (int): Top k accuracy to be determined
                 Example: k=1 -> Top 1 accuracy
                          k=5 -> Top 5 accuracy
    '''
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    '''
    Calculate the length of the caption excluding the start, end and padding tokens in the caption.
    
    Arguments:
        word_dict (dict): Dictionary of words (vocabulary)
        captions (list): List of encoded captions where each entry in the encoded caption is corresponding index from word_dict
    '''
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths

def pil_loader(path):
    '''
    Load an image from the specified path and convert to RGB
    
    Arguments:
        path (str): Complete path of the image that is to be loaded
    '''
    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def myimshow(image, ax=plt):
    '''
    Funtion to display an input image
    
    Arguments:
        image : Tensor or array of image pixel values
    '''
    
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

def generate_caption(enc_caption, word_dict):
    '''
    Function to create the caption sentence from the encoded caption using the word dictionary
    
    Arguments:
        enc_caption (list): Encoded caption in terms of dictionary indices
        word_dict (dict): Dictionary of words (vocabulary)
    '''
    
    # Using the dictionary, convert the encoded caption to normal words
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    enc_caption = enc_caption.to('cpu').tolist()
    for word_idx in enc_caption:
        if word_idx == word_dict['<start>']:
            continue
        if word_idx == word_dict['<eos>']:
            break
        sentence_tokens.append(token_dict[word_idx])
        
    # Creation of a sentence from the list of words
    caption = ''
    for word in sentence_tokens:
        if word is sentence_tokens[len(sentence_tokens) - 1]:
            caption = caption + word + '.'
        else:
            caption = caption + word + ' '
    
    return caption.capitalize()