#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
In this version I divide images to patches and apply attention on each patch separately!
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

        ########### Patch 1 #############
        self.conv11 = nn.Conv2d(128, 128, 3, 1)
        self.conv12 = nn.Conv2d(128, 256, 3, 1)
        self.conv13 = nn.Conv2d(256, 256, 3, 1)
        self.conv14 = nn.Conv2d(256, 512, 3, 1, 1)

        self.bn11 = nn.BatchNorm2d(128)
        self.bn12 = nn.BatchNorm2d(256)
        self.bn13 = nn.BatchNorm2d(256)
        self.bn14 = nn.BatchNorm2d(512)

        self.cbam11 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam12 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam13 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam14 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

        ########### Patch 2 ############
        self.conv21 = nn.Conv2d(128, 128, 3, 1)
        self.conv22 = nn.Conv2d(128, 256, 3, 1)
        self.conv23 = nn.Conv2d(256, 256, 3, 1)
        self.conv24 = nn.Conv2d(256, 512, 3, 1, 1)

        self.bn21 = nn.BatchNorm2d(128)
        self.bn22 = nn.BatchNorm2d(256)
        self.bn23 = nn.BatchNorm2d(256)
        self.bn24 = nn.BatchNorm2d(512)

        self.cbam21 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam22 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam23 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam24 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

        ########### Patch 3 ############
        self.conv31 = nn.Conv2d(128, 128, 3, 1)
        self.conv32 = nn.Conv2d(128, 256, 3, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1)
        self.conv34 = nn.Conv2d(256, 512, 3, 1, 1)

        self.bn31 = nn.BatchNorm2d(128)
        self.bn32 = nn.BatchNorm2d(256)
        self.bn33 = nn.BatchNorm2d(256)
        self.bn34 = nn.BatchNorm2d(512)

        self.cbam31 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam32 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam33 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam34 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

        ########### Patch 4 ############
        self.conv41 = nn.Conv2d(128, 128, 3, 1)
        self.conv42 = nn.Conv2d(128, 256, 3, 1)
        self.conv43 = nn.Conv2d(256, 256, 3, 1)
        self.conv44 = nn.Conv2d(256, 512, 3, 1, 1)

        self.bn41 = nn.BatchNorm2d(128)
        self.bn42 = nn.BatchNorm2d(256)
        self.bn43 = nn.BatchNorm2d(256)
        self.bn44 = nn.BatchNorm2d(512)

        self.cbam41 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam42 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam43 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam44 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

        ########### Global ############
        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv5 = nn.Conv2d(512, 512, 1, 1, 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)

        self.cbam1 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam2 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam3 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        self.cbam4 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

        # Second last, fully-connected layer related to discrete emotion labels
        self.fc_local = nn.Linear(512, 8)
        self.fc_global = nn.Linear(512, 8)
        self.fc_local_global = nn.Linear(512, 8)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, patch_11, patch_12, patch_21, patch_22):
        # Convolutional, batch-normalization and pooling layers
        # patch_11
        x_conv_branch_p11 = F.relu(self.bn11(self.conv11(patch_11)))  # 32x128x8x8
        x_conv_branch_p11, _ = self.cbam11(x_conv_branch_p11)
        x_conv_branch_p11 = F.relu(self.bn12(self.conv12(x_conv_branch_p11)))  # 32x256x6x6
        x_conv_branch_p11, _ = self.cbam12(x_conv_branch_p11)
        x_conv_branch_p11 = F.relu(self.bn13(self.conv13(x_conv_branch_p11)))  # 32x256x4x4
        x_conv_branch_p11, _ = self.cbam13(x_conv_branch_p11)
        x_conv_branch_p11 = F.relu(self.bn14(self.conv14(x_conv_branch_p11)))   # 32x512x4x4
        x_conv_branch_p11, attn_mat_p11 = self.cbam14(x_conv_branch_p11)  # attn_mat of size 32x1x4x4
        # patch_12
        x_conv_branch_p12 = F.relu(self.bn21(self.conv21(patch_12)))  # 32x128x8x8
        x_conv_branch_p12, _ = self.cbam21(x_conv_branch_p12)
        x_conv_branch_p12 = F.relu(self.bn22(self.conv22(x_conv_branch_p12)))  # 32x256x6x6
        x_conv_branch_p12, _ = self.cbam22(x_conv_branch_p12)
        x_conv_branch_p12 = F.relu(self.bn23(self.conv23(x_conv_branch_p12)))  # 32x256x4x4
        x_conv_branch_p12, _ = self.cbam23(x_conv_branch_p12)
        x_conv_branch_p12 = F.relu(self.bn24(self.conv24(x_conv_branch_p12)))  # 32x512x4x4
        x_conv_branch_p12, attn_mat_p12 = self.cbam24(x_conv_branch_p12)  # attn_mat of size 32x1x4x4
        # patch_21
        x_conv_branch_p21 = F.relu(self.bn31(self.conv31(patch_21)))  # 32x128x8x8
        x_conv_branch_p21, _ = self.cbam31(x_conv_branch_p21)
        x_conv_branch_p21 = F.relu(self.bn32(self.conv32(x_conv_branch_p21)))  # 32x256x6x6
        x_conv_branch_p21, _ = self.cbam32(x_conv_branch_p21)
        x_conv_branch_p21 = F.relu(self.bn33(self.conv33(x_conv_branch_p21)))  # 32x256x4x4
        x_conv_branch_p21, _ = self.cbam33(x_conv_branch_p21)
        x_conv_branch_p21 = F.relu(self.bn34(self.conv34(x_conv_branch_p21)))  # 32x512x4x4
        x_conv_branch_p21, attn_mat_p21 = self.cbam34(x_conv_branch_p21)  # attn_mat of size 32x1x4x4
        # patch_22
        x_conv_branch_p22 = F.relu(self.bn41(self.conv41(patch_22)))  # 32x128x8x8
        x_conv_branch_p22, _ = self.cbam41(x_conv_branch_p22)
        x_conv_branch_p22 = F.relu(self.bn42(self.conv42(x_conv_branch_p22)))  # 32x256x6x6
        x_conv_branch_p22, _ = self.cbam42(x_conv_branch_p22)
        x_conv_branch_p22 = F.relu(self.bn43(self.conv43(x_conv_branch_p22)))  # 32x256x4x4
        x_conv_branch_p22, _ = self.cbam43(x_conv_branch_p22)
        x_conv_branch_p22 = F.relu(self.bn44(self.conv44(x_conv_branch_p22)))  # 32x512x4x4
        x_conv_branch_p22, attn_mat_p22 = self.cbam44(x_conv_branch_p22)  # attn_mat of size 32x1x4x4

        x_conv_out_1 = torch.cat([x_conv_branch_p11, x_conv_branch_p11], dim=3)
        x_conv_out_2 = torch.cat([x_conv_branch_p21, x_conv_branch_p22], dim=3)
        x_conv_local_ = torch.cat([x_conv_out_1, x_conv_out_2], dim=2)

        attn_mat_1 = torch.cat([attn_mat_p11, attn_mat_p12], dim=3)
        attn_mat_2 = torch.cat([attn_mat_p21, attn_mat_p22], dim=3)
        attn_mat_local = torch.cat([attn_mat_1, attn_mat_2], dim=2)

        # Prepare features for Classification
        x_conv_local = self.global_pool(x_conv_local_)  # N x 512 x 1 x 1
        x_conv_local = x_conv_local.view(-1, 512)  # N x 512
        discrete_emotion_local = self.fc_local(x_conv_local)

        ################ Global  #####################
        x_conv_global = F.relu(self.bn1(self.conv1(x)))
        x_conv_global, _ = self.cbam1(x_conv_global)
        x_conv_global = self.pool(F.relu(self.bn2(self.conv2(x_conv_global))))
        x_conv_global, _ = self.cbam2(x_conv_global)
        x_conv_global = F.relu(self.bn3(self.conv3(x_conv_global)))
        x_conv_global, _ = self.cbam3(x_conv_global)
        x_conv_global = F.relu(self.bn4(self.conv4(x_conv_global)))
        x_conv_global_, attn_mat_global = self.cbam4(x_conv_global)  # attn_mat of size 32x1x6x6

        # Prepare features for Classification
        x_conv_global = self.global_pool(x_conv_global_)  # N x 512 x 1 x 1
        x_conv_global = x_conv_global.view(-1, 512)  # N x 512
        discrete_emotion_global = self.fc_global(x_conv_global)

        discrete_emotion_lge = discrete_emotion_local + discrete_emotion_global  # the sum of FC local and global (based on MA-Net paper)

        ########### Combined global and local ##############
        # x_conv_branch = self.bn5(x_conv_local + x_conv_global)  # check if residual block needs relu and bn?
        # x_conv_branch = x_conv_local + x_conv_global
        x_conv_combined = F.relu(self.bn5(self.conv5(x_conv_local_ + x_conv_global_)))

        # Prepare features for Classification & Regression
        x_conv_combined = self.global_pool(x_conv_combined)  # N x 512 x 1 x 1
        x_conv_combined = x_conv_combined.view(-1, 512)  # N x 512
        discrete_emotion_combined = self.fc_local_global(x_conv_combined)  # the output of FC when the local and global features are summed up!

        # Returns activations of the discrete emotion output layer and arousal and valence levels
        return discrete_emotion_local, discrete_emotion_global, discrete_emotion_combined, discrete_emotion_lge, \
               attn_mat_global, attn_mat_p11, attn_mat_p12, attn_mat_p21, attn_mat_p22

    '''def forward_to_last_conv_layer(self, x_shared_representations):
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
        return x_to_output_layer'''


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
        local_emotions = []
        global_emotions = []
        combined_emotion = []
        lge_emotion = []
        attn_head_global = []
        attn_head_11 = []
        attn_head_12 = []
        attn_head_21 = []
        attn_head_22 = []

        # Get shared representations
        x_shared_representations = self.base(x)  # 32x128x20x20

        patch_11 = x_shared_representations[:, :, 0:10, 0:10]
        patch_12 = x_shared_representations[:, :, 0:10, 10:20]
        patch_21 = x_shared_representations[:, :, 10:20, 0:10]
        patch_22 = x_shared_representations[:, :, 10:20, 10:20]

        # Add to the lists of predictions outputs from each convolutional branch in the ensemble
        for branch in self.convolutional_branches:
            discrete_emotion_local, discrete_emotion_global, discrete_emotion_combined, discrete_emotion_lge, attng, \
            attn11, attn12, attn21, attn22 = branch(x_shared_representations, patch_11, patch_12, patch_21, patch_22)
            local_emotions.append(discrete_emotion_local)
            global_emotions.append(discrete_emotion_global)
            combined_emotion.append(discrete_emotion_combined)
            lge_emotion.append(discrete_emotion_lge)
            attn_head_global.append(attng)
            attn_head_11.append(attn11)
            attn_head_12.append(attn12)
            attn_head_21.append(attn21)
            attn_head_22.append(attn22)
        attn_global = torch.stack(attn_head_global)  #.permute([1, 0, 2])  # num_branches x batch_size x H x W
        attn_11 = torch.stack(attn_head_11)
        attn_12 = torch.stack(attn_head_12)
        attn21 = torch.stack(attn_head_21)
        attn22 = torch.stack(attn_head_22)

        return local_emotions, global_emotions, combined_emotion, lge_emotion, attn_global, attn_11, attn_12, attn21, \
               attn22

