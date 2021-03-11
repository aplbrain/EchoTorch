'''
Author: Sean L. McDaniel 
Purpose: To demonstrate time series prediction using the leaky integrated fire 
echo state network 

Inpsired by: EchoTorch/examples/timeserie_prediction/narma10_esn.py
'''

'''
NEW GOAL:

* Take a look at RHN IRADs Box folder 
* Perl input network modification code into a utility function 
* Use Marisel's networks for the Reservoir 
* Randomly select some of Marisel's networks and just create figures 

NEW GOAL 3-5-21
* What plotting and eval tools we have at our disposal 
* What are our debugging tools which plots allow us to debug
* Clean and comment for others
* Send the link the Erik to play with
* Look for addtl' datasets 

NEW GOAL 3-11-21
* Create new ESN class that uses Marisol's networks instead of replacing W later
  replace at creation of ESN object
'''

import torch, sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), ''))
from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.nn.reservoir as etrs
import echotorch.nn as etnn
import echotorch.utils
import echotorch.utils.matrix_generation as mg
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def print_adj_matrix(esn):
    A = esn.w_in.cpu().numpy() # get esn wts in from cpu and conver to np array 
    reservoir_size = esn.w_in.size()[0] # get the size of the reservoir 
    result = np.zeros((reservoir_size + 1, reservoir_size + 1)) # create zeroed np array 
    reshaped_A = A.reshape((A.shape[1], A.shape[0])) # reshape A (transpose A)
    result[:reshaped_A.shape[0],:reshaped_A.shape[1]] = reshaped_A # insert A into zeroed aray
    A = np.matrix(result) # convert result matrix to numpy matrix 
    B = A != 0 # conditional to turn non zero elements to 1 instead of a float 
    C = B.astype(int) # convert from floats to ints 
    np.random.shuffle(C)

def print_adj_matrix2(esn):
    '''
    write this funtion to replace the w_in matrix with a new matrix 
    '''
    esn = esn.cpu()
    A = esn.w_in
    reservoir_size = esn.w_in.size()[0] # get the size of the reservoir 
    result = np.zeros((reservoir_size + 1, reservoir_size + 1)) # create zeroed np array 
    reshaped_A = A.reshape((A.shape[1], A.shape[0])) # reshape A (transpose A)
    result[:reshaped_A.shape[0],:reshaped_A.shape[1]] = reshaped_A # insert A into zeroed aray
    A = np.matrix(result) # convert result matrix to numpy matrix 
    B = A != 0 # conditional to turn non zero elements to 1 instead of a float 
    C = B.astype(int) # convert from floats to ints 
    print(C) # print out final Adj matrix 

def vis_weights_from_input_to_res(esn):
    A = esn.w_in.cpu().numpy() # get esn wts in from cpu and conver to np array 
    reservoir_size = esn.w_in.size()[0] # get the size of the reservoir 
    result = np.zeros((reservoir_size + 1, reservoir_size + 1)) # create zeroed np array 
    reshaped_A = A.reshape((A.shape[1], A.shape[0])) # reshape A (transpose A)
    result[:reshaped_A.shape[0],:reshaped_A.shape[1]] = reshaped_A # insert A into zeroed aray
    A = np.matrix(result) # convert result matrix to numpy matrix 
    B = A != 0 # conditional to turn non zero elements to 1 instead of a float 
    C = B.astype(int) # convert from floats to ints 
    np.savetxt('test.out', C, delimiter=',', fmt='%i') # specity int format for write out 

    G = nx.from_numpy_matrix(C, create_using=nx.DiGraph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout)
    #nx.draw_networkx_edge_labels(G, pos=layout)
    nx.draw_networkx_labels(G,pos=layout)
    plt.savefig('test.png')

# Generate the matrix
#def custom_generate_matrix(self, size, dtype=torch.float64):
def custom_generate_matrix(size, dtype=torch.float64):
    """
    Generate the matrix
    :param: Matrix size (row, column)
    :param: Data type to generate
    :return: Generated matrix
    """
    # Params
    #connectivity = self.get_parameter('connectivity')
    #mean = self.get_parameter('mean')
    #std = self.get_parameter('std')

    #connectivity = 0.1954
    connectivity = 0.0824
    mean=0.0
    std=1.0

    # Full connectivity if none
    if connectivity is None:
        w = torch.zeros(size, dtype=dtype)
        w = w.normal_(mean=mean, std=std)
    else:
        # Generate matrix with entries from norm
        w = torch.zeros(size, dtype=dtype)
        w = w.normal_(mean=mean, std=std)

        # Generate mask from bernoulli
        mask = torch.zeros(size, dtype=dtype)
        mask.bernoulli_(p=connectivity)

        # Minimum edges
        #minimum_edges = min(self.get_parameter('minimum_edges'), np.prod(size))
        minimum_edges = 0

        # Add edges until minimum is ok
        while torch.sum(mask) < minimum_edges:
            # Random position at 1
            x = torch.randint(high=size[0], size=(1, 1))[0, 0].item()
            y = torch.randint(high=size[1], size=(1, 1))[0, 0].item()
            mask[x, y] = 1.0
        # end while

        # Mask filtering
        w *= mask
    # end if

    return w
    # end _generate_matrix

# use the same distribution that that custom is using and where ever marisel is 
# a 1 pull a number from the distribution that the custom does and if its a 0
# then leave as is
def replace_res_network_matrix(esn):

    # read in HSBM network
    path = '/home/mcdansl1/Data/hsbm'
    list_of_graphml_files = os.listdir(path) # returns list
    print(list_of_graphml_files[0])
    G = nx.read_graphml(path + '/' + list_of_graphml_files[1])
    hsbm_adj_matrix = nx.to_numpy_matrix(G) # converts A's networkx to adj matrix to numpy numpy array
    print(hsbm_adj_matrix)

    mean=0.0
    std=1.0

    hsbm_adj_matrix = np.where((hsbm_adj_matrix == 1), np.random.normal(mean, std, hsbm_adj_matrix.shape), hsbm_adj_matrix)

    print(hsbm_adj_matrix)

    hsbm_adj_matrix_tensor = torch.tensor(hsbm_adj_matrix, dtype=torch.float)

    esn.w = hsbm_adj_matrix_tensor

    print(esn.w.numpy())

    return esn

def replace_res_network_matrix_with_custom(esn, matrix_tensor):
    esn.w = matrix_tensor
    return esn

def replace_input_network_matrix(esn):
    esn = esn.cpu()
    A = esn.w_in.numpy()
    reservoir_size = esn.w_in.size()[0] # get the size of the reservoir 
    result = np.zeros((reservoir_size + 1, reservoir_size + 1)) # create zeroed np array
    A = A.reshape((A.shape[1], A.shape[0])) # reshape A (transpose A)
    B = (A != 0).astype(int)
    C = np.random.randint(low=0,high=2, size=B.shape[1], dtype=int)
    D = np.zeros(B.shape, dtype=np.double)
    D[0] = C
    D_tensor = torch.tensor(D, dtype=torch.float)

    D_tensor = D_tensor.reshape((D_tensor.shape[1], D_tensor.shape[0]))
    
    esn.w_in = D_tensor

    if use_cuda:
        esn.cuda()
    
    esn = esn.cpu()

    if use_cuda:
        esn.cuda()

# Length of training samples
train_sample_length = 5500

# Length of test samples
test_sample_length = 500

# How many training/test samples
n_train_samples = 1
n_test_samples = 1

# Batch size (how many sample processed at the same time?)
batch_size = 1

# Reservoir hyper-parameters
spectral_radius = 1.07
leaky_rate = 0.9261
input_dim = 1
reservoir_size = 1000
connectivity = 0.1954
ridge_param = 0.00000409
input_scaling = 0.9252
bias_scaling = 0.079079

# Predicted/target plot length
plot_length = 200

use_cuda = False or torch.cuda.is_available()

print('Sean are we using CUDA: ', use_cuda)

# NARMA30 dataset
narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, 
    system_order=10)

narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, 
    system_order=10)

# Data loaders
trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, 
    shuffle=False, num_workers=2)

testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, 
    shuffle=False, num_workers=2)

# Internal matrix
w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity,
    spetral_radius=spectral_radius
)

# Input weights
win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity,
    scale=input_scaling,
    apply_spectral_radius=False
)

# Bias vector
wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
    connectivity=connectivity,
    scale=bias_scaling,
    apply_spectral_radius=False
)

# Create a Leaky-integrated ESN,
# with least-square training algo.
# esn = etrs.ESN(
esn = etrs.LiESN(
    input_dim=input_dim, # dims of the input used in win creation 
    hidden_dim=reservoir_size, # dims of the reservoir used in win creation 
    output_dim=1,
    leaky_rate=leaky_rate,
    learning_algo='inv',
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    ridge_param=ridge_param
)


'''
reservoir_matrix_tensor = custom_generate_matrix(
    size=(reservoir_size, reservoir_size), dtype=torch.float)

esn = replace_res_network_matrix_with_custom(esn, reservoir_matrix_tensor)
'''

esn = replace_res_network_matrix(esn)

if use_cuda:
    esn.cuda()

# For each batch
for data in trainloader:
    # Inputs and outputs
    inputs, targets = data
    
    if use_cuda: 
        inputs, targets = inputs.cuda(), targets.cuda()

    # ESN need inputs and targets
    esn(inputs, targets)
# end for

# Now we finalize the training by
# computing the output matrix Wout.
esn.finalize()

# Get the first sample in training set,
# and transform it to Variable.
dataiter = iter(trainloader)

train_inputs, train_targets = dataiter.next()

train_inputs, train_inputs = Variable(train_inputs), Variable(train_targets) 

if use_cuda: 
    train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()

print('is train x and y on gpu? ', train_inputs.is_cuda, train_targets.is_cuda)

# Make a prediction with our trained ESN
y_predicted = esn(train_inputs, None) # causing errors

# Print training MSE and NRMSE
print(u"Train MSE: {}".format(echotorch.utils.mse(y_predicted.data, 
    train_targets.data)))
print(u"Train NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, 
    train_targets.data)))

# sending tensor back to host memory so that it can be plotted
train_targets = train_targets.cpu()
y_predicted = y_predicted.cpu()

# Show target and predicted
plt.plot(train_targets[0, :plot_length, 0].data, 'r')
plt.plot(y_predicted[0, :plot_length, 0].data, 'b')
# read in HSBM network
path = '/home/mcdansl1/Data/hsbm'
list_of_graphml_files = os.listdir(path) # returns list

plt.savefig('custom.png')