#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the second version of esr_5 in which I apply the new attention methods (efficient and effective ones, SE, CBAM)
"""

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
        x_shared_representations, _ = self.cbam1(x_shared_representations)

        x_shared_representations = self.pool(F.relu(self.bn2(self.conv2(x_shared_representations))))
        x_shared_representations, _ = self.cbam2(x_shared_representations)

        x_shared_representations = F.relu(self.bn3(self.conv3(x_shared_representations)))
        x_shared_representations, _ = self.cbam3(x_shared_representations)

        x_shared_representations = self.pool(F.relu(self.bn4(self.conv4(x_shared_representations))))
        x_shared_representations, _ = self.cbam4(x_shared_representations)

        return x_shared_representations  # 32x128x20x20


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

        # Second last, fully-connected layer related to discrete emotion labels
        self.fc = nn.Linear(512, 8)

        # Last, fully-connected layer related to continuous affect levels (arousal and valence)
        self.fc_dimensional = nn.Linear(8, 2)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_shared_representations):
        # Convolutional, batch-normalization and pooling layers
        x_conv_branch = F.relu(self.bn1(self.conv1(x_shared_representations)))
        print('bl1_shape', x_conv_branch.shape)
        x_conv_branch, _ = self.cbam1(x_conv_branch)

        x_conv_branch = self.pool(F.relu(self.bn2(self.conv2(x_conv_branch))))
        print('bl2_shape', x_conv_branch.shape)
        x_conv_branch, _ = self.cbam2(x_conv_branch)

        x_conv_branch = F.relu(self.bn3(self.conv3(x_conv_branch)))
        print('bl3_shape', x_conv_branch.shape)
        x_conv_branch, _ = self.cbam3(x_conv_branch)

        x_conv_branch = F.relu(self.bn4(self.conv4(x_conv_branch)))
        print('bl4_shape', x_conv_branch.shape)
        x_conv_branch, attn_mat = self.cbam4(x_conv_branch)  # attn_mat of size 32x1x6x6

        # print('attn_head_size', x_conv_branch.shape)  # Size: 32 x 512 x 6 x 6

        # Prepare features for Classification & Regression
        x_conv_branch = self.global_pool(x_conv_branch)  # N x 512 x 1 x 1
        x_conv_branch = x_conv_branch.view(-1, 512)  # N x 512

        # Fully connected layer for expression recognition
        discrete_emotion = self.fc(x_conv_branch)

        # Application of the ReLU function to neurons related to discrete emotion labels
        x_conv_branch = F.relu(discrete_emotion)

        # Fully connected layer for affect perception
        continuous_affect = self.fc_dimensional(x_conv_branch)

        # Returns activations of the discrete emotion output layer and arousal and valence levels
        return discrete_emotion, continuous_affect, attn_mat

    def forward_to_last_conv_layer(self, x_shared_representations):
        """
        Propagates activations to the last convolutional layer of the architecture.
        This method is used to generate saliency maps with the Grad-CAM algorithm (Selvaraju et al., 2017).

        Reference:
            Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017.
            Grad-cam: Visual explanations from deep networks via gradient-based localization.
            In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

        :param x_shared_representations: (ndarray) feature maps from shared layers
        :return: feature maps of the last convolutional layer
        """

        # Convolutional, batch-normalization and pooling layers
        x_to_last_conv_layer = F.relu(self.bn1(self.conv1(x_shared_representations)))
        x_to_last_conv_layer, _ = self.cbam1(x_to_last_conv_layer)
        x_to_last_conv_layer = self.pool(F.relu(self.bn2(self.conv2(x_to_last_conv_layer))))
        x_to_last_conv_layer, _ = self.cbam2(x_to_last_conv_layer)
        x_to_last_conv_layer = F.relu(self.bn3(self.conv3(x_to_last_conv_layer)))
        x_to_last_conv_layer, _ = self.cbam3(x_to_last_conv_layer)
        x_to_last_conv_layer = F.relu(self.bn4(self.conv4(x_to_last_conv_layer)))
        x_to_last_conv_layer, _ = self.cbam4(x_to_last_conv_layer)

        # Feature maps of the last convolutional layer
        return x_to_last_conv_layer

    def forward_from_last_conv_layer_to_output_layer(self, x_from_last_conv_layer):
        """
        Propagates activations to the second last, fully-connected layer (here referred as output layer).
        This layer represents emotion labels.

        :param x_from_last_conv_layer: (ndarray) feature maps from the last convolutional layer of this branch.
        :return: (ndarray) activations of the last second, fully-connected layer of the network
        """

        # Global average polling and reshape
        x_to_output_layer = self.global_pool(x_from_last_conv_layer)
        x_to_output_layer = x_to_output_layer.view(-1, 512)

        # Output layer: emotion labels
        x_to_output_layer = self.fc(x_to_output_layer)

        # Returns activations of the discrete emotion output layer
        return x_to_output_layer


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
        self.to(device)
        # Evaluation mode on
        # self.eval()

    def get_ensemble_size(self):
        return len(self.convolutional_branches)

    def add_branch(self):
        self.convolutional_branches.append(ConvolutionalBranch())
        self.convolutional_branches[-1].to(self.device)

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
        affect_values = []
        heads = []

        # Get shared representations
        x_shared_representations = self.base(x)

        # Add to the lists of predictions outputs from each convolutional branch in the ensemble
        for branch in self.convolutional_branches:
            output_emotion, output_affect, attn_mat = branch(x_shared_representations)
            emotions.append(output_emotion)
            affect_values.append(output_affect)
            heads.append(attn_mat[:, 0, :, :])
        attn_heads = torch.stack(heads) #.permute([1, 0, 2])
        # print('attn_shape', attn_heads.shape)  # num_branches x batch_size x H x W

        return emotions, affect_values, attn_heads

