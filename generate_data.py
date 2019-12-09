'''
This file contains all the functions necessary for preprocessing of the data and generation of necessary json files that contain
image paths and captions for training, validation and testing sets. It is also responsible for creating the word dictionary (vocabulary)
for image captioning.
'''

import argparse, json
from collections import Counter

def generate_trainval_json_data(split_path, data_path, max_captions_per_image, min_word_count):
    '''
    This function is used to create json files for the image_paths and captions corresponding to the 
    training and validation sets from a split file. COCO Dataset does not have a standard split file available.
    Hence, we are using the widely used Andrej Karpathy's training and validation split.
    
    Additionally, this function is responsible for creating a dictionary of words from all the training and 
    validation captions. The words in the caption are then replaced by their key in this word dictionary so that 
    we can provide caption as a tensor to the RNN.
    
    Arguments:
        split_path (str): Complete path for the splits file. Default = 'data/coco/dataset.json'
        data_path (str): Complete path of the folder for storage of created json files. Default = 'data/coco'
        max_captions_per_image (int): Maximum number of captions per image. Default = 5
                                      This is to ensure uniformity across all images in the dataset as COCO has more than 5 captions for 
                                      some entries.
        min_word_count (int): Minimum number of occurrences of a word to be included in the dictionary. Default = 5
                              (to limit the network vocabulary)
    '''
    
    split = json.load(open(split_path, 'r'))
    word_count = Counter()

    train_img_paths = []
    train_caption_tokens = []
    val_img_paths = []
    val_caption_tokens = []

    max_length = 0
    for img in split['images']:
        caption_count = 0
        for sentence in img['sentences']:
            if caption_count < max_captions_per_image:
                caption_count += 1
            else:
                break

            if img['split'] == 'train':
                train_img_paths.append(data_path + '/imgs/' + img['filepath'] + '/' + img['filename'])
                train_caption_tokens.append(sentence['tokens'])
            elif img['split'] == 'val':
                val_img_paths.append(data_path + '/imgs/' + img['filepath'] + '/' + img['filename'])
                val_caption_tokens.append(sentence['tokens'])
            max_length = max(max_length, len(sentence['tokens']))     # to find the maximum caption length in the data
            word_count.update(sentence['tokens'])                     # update words from the caption into the dictionary counter

    words = [word for word in word_count.keys() if word_count[word] >= min_word_count]
    word_dict = {word: idx + 4 for idx, word in enumerate(words)}
    word_dict['<start>'] = 0      # indicate start of the caption
    word_dict['<eos>'] = 1        # indicate end of the caption
    word_dict['<unk>'] = 2        # indicate words that are not present in the dictionary
    word_dict['<pad>'] = 3        # for padding to create equal length captions

    with open(data_path + '/word_dict.json', 'w') as f:
        json.dump(word_dict, f)

    # Encode the captions in terms of the word dictionary keys
    train_captions = process_caption_tokens(train_caption_tokens, word_dict, max_length)
    val_captions = process_caption_tokens(val_caption_tokens, word_dict, max_length)

    with open(data_path + '/train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_path + '/val_img_paths.json', 'w') as f:
        json.dump(val_img_paths, f)
    with open(data_path + '/train_captions.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_path + '/val_captions.json', 'w') as f:
        json.dump(val_captions, f)


def process_caption_tokens(caption_tokens, word_dict, max_length):
    '''
    Function to encode the list of words in the caption into a corresponding list of their keys in the word dictionary.
    
    Arguments:
        caption_tokens (list): List of words to be processed into dictionary keys
        word_dict (dict): Dictionary of words to be used to encode list of tokens
        max_length (int): Maximum caption length
    
    Example:
        caption_tokens = ['a', 'woman', 'standing']
        word_dict = {0:'<start>', 1:'<eos>', 3:'<pad>', 6:'a', 10:'woman', 30:'standing'}
        max_length = 5
        OUT = [0, 6, 10, 30, 3, 3, 1]
    '''
    captions = []
    for tokens in caption_tokens:
        token_idxs = [word_dict[token] if token in word_dict else word_dict['<unk>'] for token in tokens]
        captions.append(
            [word_dict['<start>']] + token_idxs + [word_dict['<eos>']] +
            [word_dict['<pad>']] * (max_length - len(tokens)))

    return captions

def generate_test_json_data(test_path, data_path):
    '''
    Function to create a json file of the test image paths to assist in final visualization of captions and corresponding image.
    The image path information is obtained from the main dataset link: http://cocodataset.org/#download
    under 2014 Testing Image Info.
    
    Arguments:
        test_path (str): Complete path of the test image info json file. Default = 'data/coco/image_info_test2014.json'
        data_path (str): Complete path of the folder for storage of created json file. Default = 'data/coco'
        
    '''
    
    coco_test = json.load(open(test_path, 'r'))
    test_img_paths = {}
    
    for img,index in zip(coco_test['images'], range(len(coco_test['images']))):
        test_img_paths[index] = '/test2014/' + img['file_name']
    
    with open(data_path + '/test_img_paths.json', 'w') as f:
        json.dump(test_img_paths, f)
