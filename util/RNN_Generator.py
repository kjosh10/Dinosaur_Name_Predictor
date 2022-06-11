import numpy as np
from util.helper_functions import *

class RNN:
    """
    this script is used to create RNN units 

    Attributes:
        n_a: int, number of units of the RNN cell
        n_x: int, vocab size
        n_y: int, vocab size 
    """
    def __init__(self, n_a, n_x, n_y):
        self.__parameters = self.initialize_parameters(n_a, n_x, n_y)
        self.__gradients = None

    def get_parameters(self):
        """
        method to get the parameters

        :return parameters: dict, a dictionary of all the parameters
        """
        return self.__parameters

    def get_gradients(self):
        """
        method to get the computed gradients

        :return parameters: dict, a dictionary of  the gradients
        """
        return self.__gradients

    def initialize_parameters(self, n_a, n_x, n_y):
        """
        method to initialize parameters with small random values
        
        param: n_a: int, number of units of the RNN cell
        param: n_x: int, vocab size
        param: n_y: int, vocab size 
        return: parameters: list, python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        """
        np.random.seed(1)
        Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
        Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
        Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
        b = np.zeros((n_a, 1)) # hidden bias
        by = np.zeros((n_y, 1)) # output bias
        
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
        return parameters

    def rnn_step_forward(self, a_prev, x):
        """
        method to propogate rnn forward a single step

        :param a_prev: list, previous value of a
        :param x: np.ndarray, input x
        :return [a_next, p_t]: list, a list of relevant parameters
        """
        Waa, Wax, Wya, by, b = self.__parameters['Waa'], self.__parameters['Wax'], self.__parameters['Wya'], self.__parameters['by'], self.__parameters['b']
        # hidden state
        a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) 
        # unnormalized log probabilities for next chars # probabilities for next chars
        p_t = softmax(np.dot(Wya, a_next) + by) 
        return a_next, p_t

    def rnn_step_backward(self, dy, x, a, a_prev):
        """
        method to propogate rnn backward a single step

        :param dy: int, vocab size length
        :param x: np.ndarray, one-hot vector representation of the a character in X
        :param a: list, current value of a
        :param a_prev: list, previous value of a
        """
        self.__gradients['dWya'] += np.dot(dy, a.T)
        self.__gradients['dby'] += dy
        # backprop into h
        da = np.dot(self.__parameters['Wya'].T, dy) + self.__gradients['da_next']
        # backprop through tanh nonlinearity
        daraw = (1 - a * a) * da 
        self.__gradients['db'] += daraw
        self.__gradients['dWax'] += np.dot(daraw, x.T)
        self.__gradients['dWaa'] += np.dot(daraw, a_prev.T)
        self.__gradients['da_next'] = np.dot(self.__parameters['Waa'].T, daraw)

    def update_parameters(self, lr):
        """
        method to update parameters of the rnn

        :param lr: float, learning rate for learning the model
        """
        self.__parameters['Wax'] += -lr * self.__gradients['dWax']
        self.__parameters['Waa'] += -lr * self.__gradients['dWaa']
        self.__parameters['Wya'] += -lr * self.__gradients['dWya']
        self.__parameters['b']  += -lr * self.__gradients['db']
        self.__parameters['by']  += -lr * self.__gradients['dby']

    def rnn_forward(self, X, Y, a0, vocab_size = 27):
        """
        method for the forward propogation

        :param X: list, where each integer is a number that maps to a character in the vocabulary.
        :param Y: list, exactly the same as X but shifted one index to the left.
        :param a0: list, initial value of a
        :param vocab_size: int, vocab size length
        :return [loss, cache]: list, a list of losses and caches
        """
        # Initialize x, a and y_hat as empty dictionaries
        x, a, y_hat = {}, {}, {}
        a[-1] = np.copy(a0)
        
        # initialize your loss to 0
        loss = 0
        for t in range(len(X)):
            # Set x[t] to be the one-hot vector representation of the t'th character in X.
            # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
            x[t] = np.zeros((vocab_size,1)) 
            if (X[t] != None):
                x[t][X[t]] = 1 
            # Run one step forward of the RNN
            a[t], y_hat[t] = self.rnn_step_forward(a[t-1], x[t]) 
            # Update the loss by substracting the cross-entropy term of this time-step from it.
            loss -= np.log(y_hat[t][Y[t],0]) 
        cache = (y_hat, a, x)  
        return loss, cache

    def rnn_backward(self, X, Y, cache):
        """
        method for the backward propogation

        :param X: list, where each integer is a number that maps to a character in the vocabulary.
        :param Y: list, exactly the same as X but shifted one index to the left.
        :param cache: tuple, tuple of the cache memory
        :return a: list, a list of current a
        """
        # Initialize gradients as an empty dictionary
        self.__gradients = {}
        
        # Retrieve from cache and parameters
        (y_hat, a, x) = cache
        Waa, Wax, Wya, by, b = self.__parameters['Waa'], self.__parameters['Wax'], self.__parameters['Wya'], self.__parameters['by'], self.__parameters['b']
        
        # each one should be initialized to zeros of the same dimension as its corresponding parameter
        self.__gradients['dWax'], self.__gradients['dWaa'], self.__gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
        self.__gradients['db'], self.__gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
        self.__gradients['da_next'] = np.zeros_like(a[0])

        # Backpropagate through time
        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            self.rnn_step_backward(dy, x[t], a[t], a[t-1])
        
        return a

    def optimize(self, X, Y, a_prev, learning_rate = 0.01):
        """
        method to execute one step of the optimization to train the model
        
        :param X: list, where each integer is a number that maps to a character in the vocabulary.
        :param Y: list, exactly the same as X but shifted one index to the left.
        :param a_prev: float, previous hidden state.
        :param learning_rate: float, learning rate for the model.
        :return loss: float, value of the loss function (cross-entropy)
        :return gradients: dict, a python dictionary containing of the computed gradients
        :return a[len(X)-1]: float, the last hidden state of shape (n_a, 1)
        """
        # Forward propagate through time
        loss, cache = self.rnn_forward(X, Y, a_prev)
        # Backpropagate through time
        a = self.rnn_backward(X, Y, cache)
        # Clip your gradients between -5 (min) and 5 (max)
        self.__gradients = clip(self.__gradients, 5)
        # Update parameters 
        self.update_parameters(learning_rate)
        return loss, self.__gradients, a[len(X)-1]

    def sample(self, char_to_ix, seed):
        """
        method to sample a sequence of characters according to a sequence of probability distributions output of the RNN

        :param char_to_ix: dict, a python dictionary mapping each character to an index.
        :param seed: seed, used for grading purposes. Do not worry about it.
        :return indices: list, a list of length n containing the indices of the sampled characters.
        """
        
        # Retrieve parameters and relevant shapes from "parameters" dictionary
        Waa, Wax, Wya, by, b = self.__parameters['Waa'], self.__parameters['Wax'], self.__parameters['Wya'], self.__parameters['by'], self.__parameters['b']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]
      
        #  Create the one-hot vector x for the first character (initializing the sequence generation)
        x = np.zeros((vocab_size, 1))
        #  Initialize a_prev as zeros (≈1 line)
        a_prev = np.zeros((n_a, 1))
        
        # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate
        indices = []
        
        # Idx is a flag to detect a newline character, we initialize it to -1
        idx = -1 
        
        # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
        # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
        # trained model), which helps debugging and prevents entering an infinite loop
        counter = 0
        newline_character = char_to_ix['\n']
        
        while (idx != newline_character and counter != 50):
            # Forward propagate x
            a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
            z = np.dot(Wya, a) + by
            y = softmax(z)
            
            # for grading purposes
            np.random.seed(counter + seed) 
            
            # Sample the index of a character within the vocabulary from the probability distribution y
            idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

            # Append the index to "indices"
            indices.append(idx)
            
            # Overwrite the input character as the one corresponding to the sampled index.
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
            a_prev = a
            seed += 1
            counter +=1
        if (counter == 50):
            indices.append(char_to_ix['\n'])
        return indices

    def model(self, data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
        """
        method to train the model and generates dinosaur names

        :param data: str, a text corpus
        :param ix_to_char: dict, a dictionary that maps the index to a character
        :param char_to_ix: dict, a dictionary that maps a character to an index
        :param num_iterations: int, number of iterations to train the model for
        :param n_a: int, number of units of the RNN cell
        :param dino_names: int, number of dinosaur names you want to sample at each iteration. 
        :param vocab_size: vocab size length
        """
        # Retrieve n_x and n_y from vocab_size
        n_x, n_y = vocab_size, vocab_size
        # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
        loss = get_initial_loss(vocab_size, dino_names)
        # Build list of all dinosaur names (training examples).
        with open(".\data\dinos.txt") as f:
            examples = f.readlines()
        examples = [x.lower().strip() for x in examples]
        # Shuffle list of all dinosaur names
        np.random.seed(0)
        np.random.shuffle(examples)
        # Initialize the hidden state of your LSTM
        a_prev = np.zeros((n_a, 1))
        # Optimization loop
        for j in range(num_iterations):
            # Use the hint above to define one training example (X,Y) 
            index = j % len(examples)
            X = [None] + [char_to_ix[ch] for ch in examples[index]] 
            Y = X[1:] + [char_to_ix["\n"]]
            # Perform one optimization step: Forward-prop -> 
            # Backward-prop -> Clip -> Update parameters
            # Choose a learning rate of 0.01
            curr_loss, gradients, a_prev = self.optimize(X, Y, a_prev)
            
            # Use a latency trick to keep the loss smooth. It happens here to accelerate the training
            loss = smooth(loss, curr_loss)
            # Every 2000 Iteration, generate "n" characters thanks to sample() 
            # to check if the model is learning properly
            if j % 2000 == 0:
                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
                # The number of dinosaur names to print
                seed = 0
                for name in range(dino_names):
                    
                    # Sample indices and print them
                    sampled_indices = self.sample(char_to_ix, seed)
                    print_sample(sampled_indices, ix_to_char)
                    seed += 1  
                    # To get the same result for grading purposed, 
                    #increment the seed by one
                print('\n')