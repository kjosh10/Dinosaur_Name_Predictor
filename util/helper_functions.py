"""
importing necessary dependencies
"""

import numpy as np

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients

def softmax(x):
    """
    method to apply softmax to the input x

    :param x: np.ndarray, a numpy array
    :return np.ndarray, a numpy array of the probabilities
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss, cur_loss):
    """
    method to smooth loss

    :param loss: float, loss of the model
    :param cur_loss: float, loss of the current model
    :return float, smoothed loss
    """
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    """
    method to print the name

    :param sample_ix: list, a list of sample indices
    :param ix_to_char: list, a list of embedding from index to character
    """
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print ('%s' % (txt, ), end='')


def get_sample(sample_ix, ix_to_char):
    """
    method to get the name

    :param sample_ix: list, a list of sample indices
    :param ix_to_char: list, a list of embedding from index to character
    :return txt: str, a string of the Dinosaur name
    """
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    return txt

def get_initial_loss(vocab_size, seq_length):
    """
    method to get the initial loss

    :param vocab_size: int, vocab size length
    :param seq_length: int, sequence size length
    :return float, an initial loss
    """
    return -np.log(1.0/vocab_size)*seq_length