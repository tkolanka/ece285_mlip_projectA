"""
This script contains functions needed for visualization of the captions predicted by the network.
One of the functions showcases the step by step development of the caption along with visualization of the attention component.
This is same as the strategy used by the author to display visualizations as in the examples shown in the paper. 
The strategy used is adapted for PyTorch from here: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
The second function simply displays the image and its corresponding complete caption.
"""

import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
from math import ceil
from PIL import Image

from utils import pil_loader, data_transforms

def generate_caption_visualization(encoder, decoder, img_path, word_dict, beam_size=3):
    '''
    Function to visualize the step by step development of the caption along with the corresponding attention component visualization.
    
    Arguments:
        encoder: Instance of the trained Encoder for encoding of images
        decoder: Instance of the trained Decoder for caption prediction from encoded image
        img_path (str): Complete path of the image to be visualized
        word_dict (dict): Dictionary of words (vocabulary)
        beam_size (int): Number of top candidates to consider for beam search. Default = 3
    '''
    
    # Load the image and transform it 
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    # Get the caption and the corresponding attention weights from the trained network
    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha = decoder.caption(img_features, beam_size)

    # Using the dictionary, convert the encoded caption to normal words
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break

    # Resizing image for a standard display
    img = Image.open(img_path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) / 2
    top = (h - 224) / 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
    img = img.astype('float32') / 255

    num_words = len(sentence_tokens)
    w = np.round(np.sqrt(num_words))
    h = np.ceil(np.float32(num_words) / w)
    alpha = torch.tensor(alpha)

    # Plot the different attention weighted versions of the original image along with the resultant caption word prediction
    f = plt.figure(figsize=(8,9))
    plot_height = ceil((num_words + 3) / 4.0)
    ax1 = f.add_subplot(4, plot_height, 1)
    plt.imshow(img)
    plt.axis('off')
    for idx in range(num_words):
        ax2 = f.add_subplot(4, plot_height, idx + 2)
        label = sentence_tokens[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, label, color='black', fontsize=13)
        plt.imshow(img)

        if encoder.network == 'vgg19':
            shape_size = 14
        else:
            shape_size = 7

        alpha_img = skimage.transform.pyramid_expand(alpha[idx, :].reshape(shape_size, shape_size), upscale=16, sigma=20)
            
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

def generate_image_caption(encoder, decoder, img_path, word_dict, beam_size=3, ax=plt):
    '''
    Function to display the image along with the resultant predicted caption.
    
    Arguments:
        encoder: Instance of the trained Encoder for encoding of images
        decoder: Instance of the trained Decoder for caption prediction from encoded image
        img_path (str): Complete path of the image to be visualized
        word_dict (dict): Dictionary of words (vocabulary)
        beam_size (int): Number of top candidates to consider for beam search. Default = 3
        ax: axes for plotting
    '''
    
    # Load the image and transform it
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)
    
    # Get the caption from the trained network
    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha = decoder.caption(img_features, beam_size)
    
    # Using the dictionary, convert the encoded caption to normal words
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        if word_idx == word_dict['<start>']:
            continue
        if word_idx == word_dict['<eos>']:
            break
        sentence_tokens.append(token_dict[word_idx])
            
    # Resizing image for a standard display
    img = Image.open(img_path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) / 2
    top = (h - 224) / 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
    img = img.astype('float32') / 255
    
    # Creation of a sentence from the list of words
    caption = ''
    for word in sentence_tokens:
        if word is sentence_tokens[len(sentence_tokens) - 1]:
            caption = caption + word + '.'
        else:
            caption = caption + word + ' '
    
    ax.imshow(img)
    ax.set_title(caption.capitalize())
    ax.axis('off')