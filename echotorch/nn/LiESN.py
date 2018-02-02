# -*- coding: utf-8 -*-
#
# File : echotorch/nn/ESN.py
# Description : An Echo State Network module.
# Date : 26th of January, 2018
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

import torch
from .LiESNCell import LiESNCell
from .ESN import ESN


# Leaky-Integrated Echo State Network module
class LiESN(ESN):
    """
    Leaky-Integrated Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, spectral_radius=0.9,
                 bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, leaky_rate=1.0, train_leaky_rate=False):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param spectral_radius:
        :param bias_scaling:
        :param input_scaling:
        :param w:
        :param w_in:
        :param w_bias:
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func:
        :param learning_algo:
        :param ridge_param:
        :param leaky_rate:
        :param train_leaky_rate:
        """
        super(LiESN, self).__init__(input_dim, hidden_dim, output_dim, spectral_radius=0.9,
                                    bias_scaling=0, input_scaling=1.0,
                                    w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0],
                                    w_sparsity=None,
                                    nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, create_cell=False)

        # Recurrent layer
        self.esn_cell = LiESNCell(leaky_rate, train_leaky_rate, input_dim, hidden_dim, spectral_radius=0.9,
                                  bias_scaling=0, input_scaling=1.0,
                                  w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0],
                                  w_sparsity=None,
                                  nonlin_func=torch.tanh)
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    ###############################################
    # PUBLIC
    ###############################################

    ###############################################
    # PRIVATE
    ###############################################

# end ESNCell
