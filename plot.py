'''
This file contains functions for plotting the training and validation statistics from the log files generated.
'''

from matplotlib import pyplot as plt

def read_logfile(fname):
    '''
    This function reads the log file and extracts the training statistics, validation statistics and the bleu scores.
    '''
    
    with open(fname) as f:
        text = f.read()
        
    text = text.split('\n')
    for t in text:
        if t[:5] == 'Batch':
            text.remove(t)

    for i in range(len(text)):
        text[i] = text[i].split()

    train_data = []
    i = 0
    count = 0

    for t in text:
        if t[0] == 'Train':
            train_data.append((eval(t[2][1:-1]) + i, float(t[4]), float(t[5][1:-1]), float(t[9]), 
                               float(t[10][1:-1]), float(t[14]), float(t[15][1:-1])))
            count += 1
            if count % 130 == 0:
                i += 1

    i = 0
    count = 0
    validation_data = []
    bleu_scores = []
    for t in text:
        if t[0] == 'Validation':
            if t[1] == 'Batch:':
                validation_data.append((eval(t[2][1:-1]) + i, float(t[4]), float(t[5][1:-1]), float(t[9]), 
                                       float(t[10][1:-1]), float(t[14]), float(t[15][1:-1])))
                count += 1
                if count % 8 == 0:
                    i += 1
            elif t[1] == 'Epoch:':
                bleu_scores.append((float(t[4][1:-1]), float(t[6][1:-1]), 
                                    float(t[8][1:-1]), float(t[10][1:-1])))
    return train_data, validation_data, bleu_scores

def trim_data(data):
    '''
    This function selects a subset of the logged training statistics data for plotting.
    '''
    count = 0
    final = []
    final.append(data[0])
    for i in data:
        count += 1
        if count % 4 == 0:
            final.append(i)
    return final

def plot(fig, axes, data, data_type, ylim):
    '''
    Main plot function for displaying the trend in the loss, top 1 accuracy and top 5 accuracy.
    '''
    axes[0].clear()
    axes[1].clear()
    axes[2].clear()
    
    x_axis = [a for a,_,_,_,_,_,_ in data]
    abs_loss = [a for _,a,_,_,_,_,_ in data]
    abs_top1 = [a for _,_,_,a,_,_,_ in data]
    abs_top5 = [a for _,_,_,_,_,a,_ in data]
    
    avg_loss = [a for _,_,a,_,_,_,_ in data]
    avg_top1 = [a for _,_,_,_,a,_,_ in data]
    avg_top5 = [a for _,_,_,_,_,_,a in data]
    
    
    axes[0].plot(x_axis, abs_loss, label="Absolute", alpha=0.4)
    axes[0].plot(x_axis, avg_loss, label="Average")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(ylim[0])
    axes[0].legend()
    axes[0].set_title('{} Loss'.format(data_type))
    axes[1].plot(x_axis, abs_top1, label="Absolute", alpha=0.4)
    axes[1].plot(x_axis, avg_top1, label="Average")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(ylim[1])
    axes[1].legend()
    axes[1].set_title('{} Top 1 Accuracy'.format(data_type))
    axes[2].plot(x_axis, abs_top5, label="Absolute", alpha=0.4)
    axes[2].plot(x_axis, avg_top5, label="Average")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(ylim[2])
    axes[2].legend()
    axes[2].set_title('{} Top 5 Accuracy'.format(data_type))
    plt.tight_layout()
    fig.canvas.draw()

def processing_logfile(fname):
    train, validate, bleu = read_logfile(fname)
    train = trim_data(train)
    return train, validate, bleu