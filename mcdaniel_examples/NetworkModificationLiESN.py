'''
Author: Sean L. McDaniel 
Purpose: To demonstrate time series prediction using the leaky integrated fire 
echo state network 

Inpsired by: EchoTorch/examples/timeserie_prediction/narma10_esn.py
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


def replace_adj_matrix(esn):
    esn = esn.cpu()
    #print(esn.w_in.dtype)
    #print(esn.w_in.shape)
    #print(esn.w_in)
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
    #print(D_tensor)
    #print(D_tensor.shape)
    esn.w_in = D_tensor

    if use_cuda:
        esn.cuda()
    
    esn = esn.cpu()
    #print(esn.w_in.dtype)

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
reservoir_size = 10
connectivity = 0.1954
ridge_param = 0.00000409
input_scaling = 0.9252
bias_scaling = 0.079079

# Predicted/target plot length
plot_length = 200

use_cuda = False or torch.cuda.is_available()

print('Sean are we using CUDA: ', use_cuda)

# Manual seed initialisation
#np.random.seed(1)
#torch.manual_seed(1)


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

if use_cuda:
    esn.cuda()

#print_adj_matrix(esn)
#print_adj_matrix2(esn)
#vis_weights_from_input_to_res(esn)
replace_adj_matrix(esn)

#'''
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
plt.savefig('netmod2.png')
#'''




