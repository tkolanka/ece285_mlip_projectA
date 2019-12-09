'''
This file contains the class definitions for the basic modules that will be utilized for implementing the show attend and tell network architecture. These modules are encoder (cnn network), attention (deterministic soft attention) and decoder (rnn - lstm cell network).
'''

import torch
import torch.nn as nn
from torchvision.models import resnet152, vgg19, resnet50
import random

class Encoder(nn.Module):
    '''
    A class to load a pre-trained encoder that performs feature extraction on the input image.
    Features are extracted from the convolutional layer of the network instead of the fully connected layer,
    so as to focus on parts of image and sub-select features.
    
    Arguments:
        network (str): Network to be used as the encoder. Options = 'vgg19', 'resnet152' or 'resnet50'. Default = 'vgg19'
    '''
    
    def __init__(self, network='vgg19'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            # Remove linear and pool layers as classification is not being performed
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'resnet50':
            self.net = resnet50(pretrained=True)
            # Remove linear and pool layers as classification is not being performed
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'vgg19':
            self.net = vgg19(pretrained=True)
            # Remove the final max pooling layer to get the output of final convolutional layer
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512

    def forward(self, x):
        '''
        Forward propogation through the encoder.
        
        Arguments:
            x (tensor): Input image (batch_size, num_channels = 3, image_size = (224, 224))
        
        Output:
            x (tensor): Extracted features (batch_size, feature_pixels = 14x14, encoder.dim=512)
        '''
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x
    

class Attention(nn.Module):
    '''
    A class that implements the attention module that is to be included in the decoder for a show, attend and tell implementation.
    The current implementation is a deterministic soft attention model. This is smooth and differentiable.
    Thus, end to end learning is possible using backpropogation.
    
    Arguments:
        encoder_dim (int): The output dimension of the encoder. Default = 512
    '''
    
    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512) # Linear layer to transform decoder's earlier output
        self.W = nn.Linear(encoder_dim, 512)  # Linear layer to transform the encoded image features
        self.v = nn.Linear(512, 1)  
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)  # Computation of the weights - alpha

    def forward(self, img_features, hidden_state):
        '''
        Forward propogation through attention model
        
        Arguments:
            img_features (tensor): Extracted features from the encoder module. (batch_size, feature_pixels = 14x14, encoder_dim = 512)
            hidden_state (tensor): Previous iteration decoder output. (batch_size, decoder_dim = 512)
            
        Output:
            weighted_img_features (tensor): Extracted features weighted based on current attention (batch_size, encoder_dim = 512)
            alpha (tensor): Weights for the current attention (batch_size, feature_pixels= 14x14)
        '''
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(U_h + W_s)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        weighted_img_features = (img_features * alpha.unsqueeze(2)).sum(1)
        return weighted_img_features, alpha
    
class Decoder(nn.Module):
    '''
    A class that implements the decoder module with attention for the show, attend and tell implementation.
    The decoder is a recurrent neural network and the main unit in it is the LSTMCell. As we need to insert the attention component, the
    network is created using LSTMCell with an external for loop instead of using LSTM module.
    
    This decoder implements Scheduled Sampling for the implementation of Teacher Forcing. This means, based on the provided teacher forcing 
    ratio, teacher forcing will only be utilized for a random set of batches during training while the remaining will train
    without teacher forcing. This is done to attain a tradeoff between the pros and cons of using teacher forcing.
    
    Arguments:
        device (str): 'cuda' or 'cpu' based on which device is available
        vocabulary_size (int): Number of words in the word_dict (vocabulary of the network). Default = len(word_dict)
        encoder_dim (int): The output dimension of the encoder. Default = 512
        tf_ratio (float): Teacher forcing ratio (must lie between 0 and 1). Default = 0
                          tf_ratio = 0 --> Teacher forcing will be always used
                          tf_ratio = 1 --> Teacher forcing will never be used
    '''
    
    def __init__(self, device, vocabulary_size, encoder_dim, tf_ratio=0):
        super(Decoder, self).__init__()
        self.tf_ratio = tf_ratio
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512) # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, 512) # linear layer to find initial cell state of LSTMCell
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, encoder_dim) # Linear layer to create a sigmoid activated gate
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, vocabulary_size) # linear layer to find scores over the entire vocabulary
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim) # attention module
        self.embedding = nn.Embedding(vocabulary_size, 512)  # embedding layer
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)  # decoding LSTMCell

    def forward(self, img_features, captions):
        """
        Forward propogation through the decoder network.
        
        Arguments:
            img_features (tensor): Extracted features from the encoder module. (batch_size, feature_pixels = 14x14, encoder_dim = 512)
            captions (tensor): Captions encoded as keys of the word dictionary. (batch_size, max_caption_length)
            
        Output:
            preds (tensor): Prediction scores over the entire vocabulary
            alphas (tensor): Weights for the current attention
            batch_tf (bool): True if teach forcing was used in the current iteration, else False
        """
        
        batch_size = img_features.size(0)
        
        # Get the initial LSTM state
        h, c = self.get_init_lstm_state(img_features)
        
        # As we are using a single LSTMCell, in order to generate the entire caption we need to iterate maximum caption length number of
        # times to generate the complete predicted caption
        max_timespan = max([len(caption) for caption in captions]) - 1
        
        # Determine whether teacher forcing is to be used for the current batch or not based on the provided teacher forcing ratio
        batch_tf = True if random.random() > self.tf_ratio else False

        prev_words = torch.zeros(batch_size, 1).long().to(self.device)
        # If teacher forcing is to be used, then the ideal output is provided to the network, otherwise the previous output of the network 
        # is given as input
        if batch_tf and self.training:
            embedding = self.embedding(captions)
        else:
            embedding = self.embedding(prev_words)

        # Create tensors to hold prediction scores and alpha - weights
        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(self.device)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(self.device)
        
        # At each time-step, decode by attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            if batch_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training or not batch_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
        return preds, alphas, batch_tf

    def get_init_lstm_state(self, img_features):
        '''
        Function to get the initial hidden state and cell state of LSTM based on encoded images.
        
        Arguments:
            img_features (tensor): Extracted features from the encoder module. (batch_size, feature_pixels = 14x14, encoder_dim = 512)
        '''
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c

    def caption(self, img_features, beam_size):
        """
        Function to generate the caption for the corresponding encoded image using beam search to provide the most optimal caption 
        combination. This function is useful during human evaluation of the decoder to assess the quality of produced captions and while 
        producing visualizations of attention and corresponding produced words.
        
        Arguments:
            img_features (tensor): Extracted features from the encoder module. (batch_size, feature_pixels = 14x14, encoder_dim = 512)
            beam_size (int): Number of top candidates to consider for beam search. Default = 3
            
        Output:
            sentence (list): ordered list of words of the final optimal caption
            alpha (tensor): weights corresponding to the generated caption
        """
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha
