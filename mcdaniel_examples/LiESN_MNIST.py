'''
Author: Sean L. McDaniel 
Purpose: To demonstrate time series prediction using the leaky integrated fire 
echo state network 

Inpsired by: EchoTorch/examples/timeserie_prediction/narma10_esn.py

Current work in progress!!!!
Note: ESNJS works well but im trying ot get this to work also

Current issue:
RuntimeError: size mismatch, m1: [101 x 1500], m2: [100 x 10] at /opt/conda/conda-bld/pytorch_1570910687650/work/aten/src/THC/generic/THCTensorM
'''

# Imports
import sys, os
import torch.utils.data
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), ''))
from torch.autograd import Variable
import torchvision.datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
#from modules import ESNJS
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import Pdb
import echotorch.nn as etnn
import echotorch.nn.reservoir
import echotorch.nn.reservoir as etrs
from IPython.core.debugger import Pdb

# Experiment parameters

# General Parameters 
degrees = [30, 60, 60]
block_size = 100
training_size = 60000
test_size = 10000
n_digits = 10
batch_size = 100
use_cuda = False or torch.cuda.is_available() 

# ESN parameters
reservoir_size = 100
connectivity = 0.1
spectral_radius = 1.3
leaky_rate = 0.2
input_scaling = 0.6
ridge_param = 0.0
bias_scaling = 1.0
image_size = 15 # this is used for image deformation and was a tunable parameter in the ET MNIST paper
input_size = (len(degrees) + 1) * image_size

# MNIST data set train
# Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(
    # Transform a dataset of images into timeseries
    echotorch.datasets.ImageToTimeseries(
        # Doing some black magic with the MNIST data that comes with EchoTorch
        torchvision.datasets.MNIST(
            root=".", # Root directory of dataset where MNIST/processed/training.pt
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                echotorch.transforms.images.Concat([
                    echotorch.transforms.images.CropResize(size=image_size),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[0]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[1]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[2]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ])
                ],
                    sequential=True
                ),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=echotorch.transforms.targets.ToOneHot(class_size=n_digits)
        ), # end of torchvision.datasets.MNIST()
        n_images=block_size
    ), # end of ImageToTimeseries()
    batch_size=batch_size,
    shuffle=False
) # end of data loader 

# MNIST data set test
test_loader = torch.utils.data.DataLoader(
    echotorch.datasets.ImageToTimeseries(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                echotorch.transforms.images.Concat([
                    echotorch.transforms.images.CropResize(size=image_size),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[0]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[1]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ]),
                    torchvision.transforms.Compose([
                        echotorch.transforms.images.Rotate(degree=degrees[2]),
                        echotorch.transforms.images.CropResize(size=image_size)
                    ])
                ],
                    sequential=True
                ),
                torchvision.transforms.ToTensor()
            ]),
            target_transform=echotorch.transforms.targets.ToOneHot(class_size=n_digits)
        ),
        n_images=block_size
    ),
    batch_size=batch_size,
    shuffle=False
)

# Internal matrix
# Creates the interal resevoir matrix 
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

esn = etrs.LiESN(
    input_dim=input_size,
    hidden_dim=reservoir_size,
    output_dim=10, # we want an output for each MNIST digit
    leaky_rate=leaky_rate,
    w_generator=w_generator,
    win_generator=win_generator,
    wbias_generator=wbias_generator,
    ridge_param=ridge_param
)

if use_cuda:
    esn.cuda()
    print('GPU present, running on GPU')

# For each training sample
# Using tqdm to create a progress bar as pbar
with tqdm(total=training_size) as pbar:
    #Pdb().set_trace()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # torch.Size([10, 1, 1500, 60])
        
        # Remove channel
        data = data.reshape(batch_size, 1500, 60) # data is a tensor 
        
        # torch.Size([10, 1500, 60])
        
        # To Variable
        inputs, targets = Variable(data.float()), Variable(targets.float())
        
        #inputs, targets = data.double(), targets.double()
        # CUDA
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # end if

        # Feed ESN
        # This calls the forward function in the ESN class 
        # esn inputs and targets returns the trained states 
        states = esn(inputs, targets)
        print(states)
        # Update bar
        pbar.update(batch_size * block_size)   
    # end for
# end