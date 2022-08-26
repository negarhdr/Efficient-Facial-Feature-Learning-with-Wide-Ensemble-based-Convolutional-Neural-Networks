#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"

# Standard libraries
import torch.nn.functional as F
import torch.nn as nn
import torch
from os import path, makedirs
import copy
# External modules
from .cbam import CBAM
from torch.autograd import Variable


class Base(nn.Module):
    """
        The base of the network (Ensembles with Shared Representations, ESRs) is responsible for learning low- and
        mid-level representations from the input data that are shared with an ensemble of convolutional branches
        on top of the architecture.

        In our paper (Siqueira et al., 2020), it is called shared layers or shared representations.
    """

    def __init__(self):
        super(Base, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # Attention layers
        self.cbam1 = CBAM(gate_channels=64, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam2 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam3 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam4 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Convolutional, batch-normalization and pooling layers for representation learning
        x_shared_representations = F.relu(self.bn1(self.conv1(x)))
        x_shared_representations, _, _ = self.cbam1(x_shared_representations)

        x_shared_representations = self.pool(F.relu(self.bn2(self.conv2(x_shared_representations))))
        x_shared_representations, _, _ = self.cbam2(x_shared_representations)

        x_shared_representations = F.relu(self.bn3(self.conv3(x_shared_representations)))
        x_shared_representations, _, _ = self.cbam3(x_shared_representations)

        x_shared_representations = self.pool(F.relu(self.bn4(self.conv4(x_shared_representations))))
        x_shared_representations, _, _ = self.cbam4(x_shared_representations)

        return x_shared_representations


class ConvolutionalBranch(nn.Module):
    """
        Convolutional branches that compose the ensemble in ESRs. Each branch was trained on a sub-training
        set from the AffectNet dataset to learn complementary representations from the data (Siqueira et al., 2020).

        Note that, the second last layer provides eight discrete emotion labels whereas the last layer provides
        continuous values of arousal and valence levels.
    """

    def __init__(self):
        super(ConvolutionalBranch, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.cbam1 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam2 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam3 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam4 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        dtype = torch.FloatTensor
        self.fc_weight = Variable(torch.randn(512, 8).type(dtype), requires_grad=True).to('cuda')
        self.fc_bias = Variable(torch.randn(8).type(dtype), requires_grad=True).to('cuda')

    def forward(self, x_shared_representations):
        # Convolutional, batch-normalization and pooling layers
        x_conv_branch = F.relu(self.bn1(self.conv1(x_shared_representations)))
        x_conv_branch, _, _ = self.cbam1(x_conv_branch)

        x_conv_branch = self.pool(F.relu(self.bn2(self.conv2(x_conv_branch))))
        x_conv_branch, _, _ = self.cbam2(x_conv_branch)

        x_conv_branch = F.relu(self.bn3(self.conv3(x_conv_branch)))
        x_conv_branch, _, _ = self.cbam3(x_conv_branch)

        x_conv_branch = F.relu(self.bn4(self.conv4(x_conv_branch)))
        x_conv_branch, attn_sp, attn_ch = self.cbam4(x_conv_branch)  # attn_mat of size 32x1x6x6

        # print('attn_head_size', x_conv_branch.shape)  # Size: 32 x 512 x 6 x 6

        # Prepare features for Classification & Regression
        x_conv_branch = self.global_pool(x_conv_branch)  # N x 512 x 1 x 1
        x_conv_branch = x_conv_branch.view(-1, 512)  # N x 512

        # emotion classification
        emotions = []
        print('x_conv_branch', x_conv_branch.is_cuda)
        print('self.fc_weight', self.fc_weight.is_cuda)
        out_global = torch.mm(x_conv_branch, self.fc_weight) + self.fc_bias
        emotions.appendd(out_global)
        for i in range(8):
            l = i * 64
            u = (i + 1) * 64
            out_local = torch.mm(x_conv_branch[:, l:u], self.fc_weight[l:u, :]) + self.fc_bias
            emotions.append(out_local)

        # Returns activations of the discrete emotion output layer and arousal and valence levels
        return emotions, attn_sp  # x_conv_branch


'''class Classifier(nn.Module):
    """
        Convolutional branches that compose the ensemble in ESRs. Each branch was trained on a sub-training
        set from the AffectNet dataset to learn complementary representations from the data (Siqueira et al., 2020).

        Note that, the second last layer provides eight discrete emotion labels whereas the last layer provides
        continuous values of arousal and valence levels.
    """

    def __init__(self):
        super(Classifier, self).__init__()

        # Convolutional layers
        self.fc = nn.Linear(512, 8)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(64, 8)
        self.fc5 = nn.Linear(64, 8)
        self.fc6 = nn.Linear(64, 8)
        self.fc7 = nn.Linear(64, 8)
        self.fc8 = nn.Linear(64, 8)

    def forward(self, x):
        out = self.fc(x)
        out1 = self.fc1(x[:, 0:64])
        out2 = self.fc2(x[:, 64:128])
        out3 = self.fc3(x[:, 128:192])
        out4 = self.fc4(x[:, 192:256])
        out5 = self.fc5(x[:, 256:320])
        out6 = self.fc6(x[:, 320:384])
        out7 = self.fc7(x[:, 384:448])
        out8 = self.fc8(x[:, 448:512])
        return [out, out1, out2, out3, out4, out5, out6, out7, out8]'''


class ESR(nn.Module):
    """
    ESR is the unified ensemble architecture composed of two building blocks the Base and ConvolutionalBranch
    classes as described below by Siqueira et al. (2020):

    'An ESR consists of two building blocks. (1) The base (class Base) of the network is an array of convolutional
    layers for low- and middle-level feature learning. (2) These informative features are then shared with
    independent convolutional branches (class ConvolutionalBranch) that constitute the ensemble.'
    """

    # Default values
    # Input size
    INPUT_IMAGE_SIZE = (96, 96)
    # Values for pre-processing input data
    INPUT_IMAGE_NORMALIZATION_MEAN = [0.0, 0.0, 0.0]
    INPUT_IMAGE_NORMALIZATION_STD = [1.0, 1.0, 1.0]
    # Path to saved network
    PATH_TO_SAVED_NETWORK = "./model/ml/trained_models/esr_9_cbam"
    FILE_NAME_BASE_NETWORK = "Net-Base-Shared_Representations.pt"
    FILE_NAME_CONV_BRANCH = "Net-Branch_{}.pt"

    def __init__(self, device):
        """
        Loads ESR-9.

        :param device: Device to load ESR-9: GPU or CPU.
        """

        super(ESR, self).__init__()

        # Base of ESR-9 as described in the docstring (see mark 1)
        self.base = Base()
        self.device = device
        self.base.to(self.device)

        # Load 9 convolutional branches that composes ESR-9 as described in the docstring (see mark 2)
        self.convolutional_branches = []
        # self.classifiers = []
        self.to(device)
        # Evaluation mode on
        # self.eval()

    def get_ensemble_size(self):
        return len(self.convolutional_branches)

    def add_branch(self):
        self.convolutional_branches.append(ConvolutionalBranch())
        self.convolutional_branches[-1].to(self.device)
        # self.classifiers.append(Classifier())
        # self.classifiers[-1].to(self.device)

    @staticmethod
    def save(state_dicts, base_path_to_save_model, current_branch_save):
        if not path.isdir(path.join(base_path_to_save_model, str(current_branch_save))):
            makedirs(path.join(base_path_to_save_model, str(current_branch_save)))

        torch.save(state_dicts[0],
                   path.join(base_path_to_save_model, str(current_branch_save),
                             "Net-Base-Shared_Representations.pt"))

        for i in range(1, len(state_dicts)):
            torch.save(state_dicts[i], path.join(base_path_to_save_model, str(current_branch_save),
                                                 "Net-Branch_{}.pt".format(i)))

        print("Network has been "
              "saved at: {}".format(path.join(base_path_to_save_model, str(current_branch_save))))

    def to_state_dict(self):
        state_dicts = [copy.deepcopy(self.base.state_dict())]
        for b in self.convolutional_branches:
            state_dicts.append(copy.deepcopy(b.state_dict()))

        return state_dicts

    def to_device(self, device_to_process="cpu"):
        self.to(device_to_process)
        self.base.to(device_to_process)

        for b_td in self.convolutional_branches:
            b_td.to(device_to_process)

    def load(self, device, ensemble_size=9):
        # load base
        self.base.load_state_dict(torch.load(path.join(ESR.PATH_TO_SAVED_NETWORK, ESR.FILE_NAME_BASE_NETWORK),
                                             map_location=device))
        self.base.to(device)
        # load branches
        for i in range(1, ensemble_size + 1):
            self.convolutional_branches.append(ConvolutionalBranch())
            self.convolutional_branches[-1].load_state_dict(torch.load(
                path.join(ESR.PATH_TO_SAVED_NETWORK, ESR.FILE_NAME_CONV_BRANCH.format(i)), map_location=device))
            self.convolutional_branches[-1].to(device)
        self.to(device)

    def reload(self, best_configuration):
        self.base.load_state_dict(best_configuration[0])

        for i in range(self.get_ensemble_size()):
            self.convolutional_branches[i].load_state_dict(best_configuration[i + 1])

    def forward(self, x):
        """
        Forward method of ESR-9.

        :param x: (ndarray) Input data.
        :return: A list of emotions and affect values from each convolutional branch in the ensemble.
        """

        # List of emotions and affect values from the ensemble
        emotions = []
        heads_sp = []
        local_feat = []

        # Get shared representations
        x_shared_representations = self.base(x)
        # Add to the lists of predictions outputs from each convolutional branch in the ensemble
        for branch in self.convolutional_branches:
            classified_outs, attn_sp = branch(x_shared_representations)
            emotions.append(classified_outs)
            # local_feat.append()
            heads_sp.append(attn_sp[:, 0, :, :])
        attn_heads_sp = torch.stack(heads_sp)
        return emotions, attn_heads_sp


'''for i in range(len(self.convolutional_branches)):
    branch_features, attn_sp = self.convolutional_branches[i](x_shared_representations)
    classified_outs = self.classifiers[i](branch_features)
    emotions.append(classified_outs)
    heads_sp.append(attn_sp[:, 0, :, :])
attn_heads_sp = torch.stack(heads_sp)'''


