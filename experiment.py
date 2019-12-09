'''
This file describes the Experiment class that can be used to create different experiments for the show, attend and tell networks training and validation. This experiment is instantiated and ran in different jupyter notebooks and the performance is analyzed consequently.
For validating the performance of the network, the BLEU scores are used.
'''

import json
import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from models import Encoder, Decoder
from utils import AverageMeter, accuracy, calculate_caption_lengths


class Experiment(object):
    '''
    A class for creating an experiment of the show, attend and tell network. The experiment stores the main architecture of the network and 
    its train and validate functions can be used to run the experiment over required number of epochs. Additonally, it is possible to
    store/load the state of the model in the experiment between epochs.
    
    Arguments:
        start_epoch (int): Epoch to start the experiment from
        encoder : Instance of the Encoder class for conversion of input image to encoded features
        decoder: Instance of the Decoder class for caption prediction from the encoded features
        optimizer: Instance of the optimizer object in pytorch to make updates based on the gradients
        loss_func: Loss to be minimized during network training
        train_loader, val_loader: Instance of Dataloader to get batches of input images for train and validation
        word_dict (dict): Dictionary of the prominent words used for caption encoding
        alpha_c (float): Regularization constant. Default = 1.
        log_file (file handle): File handle for logging training and validation statistics
        log_interval (int): Frequency of logging statistics - signifies number of batches
        device (str): 'cuda' or 'cpu' based on availability
    '''
    
    def __init__(self, start_epoch, encoder, decoder, optimizer, loss_func, train_loader, val_loader, word_dict, alpha_c, log_file, log_interval, device): 
        self.start_epoch = start_epoch
        
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss = loss_func
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.word_dict = word_dict
        self.alpha_c = alpha_c
        
        self.log_file = log_file
        self.log_interval = log_interval
        
        self.device = device
        
    def load(self, model):
        '''
        Function to load from a previous model state (last checkpoint)
        
        Arguments:
            model (str): Complete path of the file that contains the previous model state (checkpoint)
        '''
        if model is not None:
            checkpoint = torch.load(model)
            self.start_epoch = checkpoint['epoch']
            self.decoder.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.log_file.write(f'Model loaded from checkpoint: {model}')
        
    def save(self, epoch, model):
        '''
        Function to save the current model state namely that of decoder and optimizer (last checkpoint).
        
        Arguments:
            epoch (int): Current epoch number
            model (str): Complete path of the file to store the state in
        '''
        
        torch.save({
          'epoch': epoch + 1,
          'state_dict': self.decoder.state_dict(),
          'optimizer' : self.optimizer.state_dict()},model)
        self.log_file.write(f'Model saved at: {model}')
        
    def train(self):
        '''
        Function for training of the network for one epoch.
        '''
        
        # Encoder is not to be trained hence it is set in evaluation mode
        self.encoder.eval()
        self.decoder.train()

        # Meters to monitor the loss, top1 accuracy and top5 accuracy
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # Iterate over all batches in the dataset to train the network
        for batch_idx, (imgs, captions) in enumerate(self.train_loader):
            imgs, captions = Variable(imgs).to(self.device), Variable(captions).to(self.device)
            
            # Images are encoded into feature maps
            img_features = self.encoder(imgs)
            self.optimizer.zero_grad()
            
            # Determine the predicted caption from the decoder network
            preds, alphas, batch_tf = self.decoder(img_features, captions)
            targets = captions[:, 1:]

            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
            
            # Determine the loss from the current prediction and using backpropogation update the decoder based on this loss
            att_regularization = self.alpha_c * ((1 - alphas.sum(1))**2).mean()
            loss = self.loss(preds, targets)
            loss += att_regularization
            loss.backward()
            self.optimizer.step()

            # Update the loss and accuracy in the corresponding trackers
            total_caption_length = calculate_caption_lengths(self.word_dict, captions)
            acc1 = accuracy(preds, targets, 1)
            acc5 = accuracy(preds, targets, 5)
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            # Based on the batch number, log the training statistics
            if batch_idx % self.log_interval == 0:
                self.log_file.write('Train Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                          batch_idx, len(self.train_loader), loss=losses, top1=top1, top5=top5))
                self.log_file.write(f'Batch Using Teacher Forcing: {batch_tf}\n')
                print('Train Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                          batch_idx, len(self.train_loader), loss=losses, top1=top1, top5=top5))
                print(f'Batch Using Teacher Forcing: {batch_tf}')
            
            torch.cuda.empty_cache()
            
    def validate(self, epoch):
        '''
        Funtion to validate the performance of the current version of the network.
        
        Arguments:
            epoch (int): Current epoch
        '''
        
        # For validation, both encoder and decoder to be used in the evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # Meters to monitor the loss, top1 accuracy and top5 accuracy
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # Required for calculation of bleu scores
        references = []
        hypotheses = []
        
        # Network to be evaluated without tracking the gradients as we are in validation mode
        with torch.no_grad():
            # Iterate over all batches in the dataset to evaluate network performance
            for batch_idx, (imgs, captions, all_captions) in enumerate(self.val_loader):
                imgs, captions = Variable(imgs).to(self.device), Variable(captions).to(self.device)
                
                # Images are encoded into feature maps
                img_features = self.encoder(imgs)
                
                # Determine the predicted caption from the decoder network
                preds, alphas, batch_tf = self.decoder(img_features, captions)
                targets = captions[:, 1:]

                targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
                packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

                # Determine the loss from the current prediction
                att_regularization = self.alpha_c * ((1 - alphas.sum(1))**2).mean()
                loss = self.loss(packed_preds, targets)
                loss += att_regularization

                # Update the loss and accuracy in the corresponding trackers
                total_caption_length = calculate_caption_lengths(self.word_dict, captions)
                acc1 = accuracy(packed_preds, targets, 1)
                acc5 = accuracy(packed_preds, targets, 5)
                losses.update(loss.item(), total_caption_length)
                top1.update(acc1, total_caption_length)
                top5.update(acc5, total_caption_length)

                # Reference (target) captions for the current image are updated in the references list
                for cap_set in all_captions.tolist():
                    caps = []
                    for caption in cap_set:
                        cap = [word_idx for word_idx in caption
                                        if word_idx != self.word_dict['<start>'] and word_idx != self.word_dict['<pad>']]
                        caps.append(cap)
                    references.append(caps)
    
                # Predicted caption is updated in the hypotheses list
                word_idxs = torch.max(preds, dim=2)[1]
                for idxs in word_idxs.tolist():
                    hypotheses.append([idx for idx in idxs
                                           if idx != self.word_dict['<start>'] and idx != self.word_dict['<pad>']])

                # Based on the batch number, log the training statistics
                if batch_idx % self.log_interval == 0:
                    self.log_file.write('Validation Batch: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                              batch_idx, len(self.val_loader), loss=losses, top1=top1, top5=top5))

            # After complete validation for the current epoch, report the four BLEU scores
            bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
            bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
            bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = corpus_bleu(references, hypotheses)

            self.log_file.write('Validation Epoch: {}\t'
                  'BLEU-1 ({})\t'
                  'BLEU-2 ({})\t'
                  'BLEU-3 ({})\t'
                  'BLEU-4 ({})\t\n'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))
            print('Validation Epoch: {}\t'
                  'BLEU-1 ({})\t'
                  'BLEU-2 ({})\t'
                  'BLEU-3 ({})\t'
                  'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))