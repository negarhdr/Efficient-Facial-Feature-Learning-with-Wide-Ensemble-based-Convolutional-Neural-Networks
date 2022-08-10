
# External libraries
import torch.nn.functional as F
import torch.nn as nn
import torch
from os import path, makedirs
import copy
from .cbam import CBAM



class BRegNextShortcutModifier(torch.nn.Module):

    def __init__(self,):
        super(BRegNextShortcutModifier, self).__init__()

        self._a = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self._c = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, inputs):
        numl = torch.atan((self._a * inputs) / torch.sqrt(self._c ** 2 + 1))
        denom = self._a * torch.sqrt(self._c ** 2 + 1)
        return numl/denom


class BReGNeXtResidualLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, downsample_stride = 1):
        super(BReGNeXtResidualLayer, self).__init__()

        self._out_channels = out_channels
        self._in_channels = in_channels
        self._downsample_stride = downsample_stride

        self._conv0 = torch.nn.Conv2d(in_channels, out_channels, 3, downsample_stride)
        torch.nn.init.kaiming_uniform_(self._conv0.weight)
        self._conv1 = torch.nn.Conv2d(out_channels, out_channels, 3, 1)
        torch.nn.init.kaiming_uniform_(self._conv1.weight)
        self._shortcut = BRegNextShortcutModifier()
        self._batchnorm_conv0 = torch.nn.BatchNorm2d(self._in_channels)
        self._batchnorm_conv1 = torch.nn.BatchNorm2d(self._out_channels)

    def forward(self, inputs):
        # First convolution
        normed_inputs = inputs if self._batchnorm_conv0 is None else self._batchnorm_conv0(inputs)
        normed_inputs = torch.nn.functional.elu(normed_inputs)
        normed_inputs = torch.nn.functional.pad(normed_inputs, (1, 1, 1, 1, 0, 0))
        conv0_outputs = self._conv0(normed_inputs)

        # Second convolution
        normed_conv0_outputs = conv0_outputs if self._batchnorm_conv1 is None else self._batchnorm_conv1(conv0_outputs)
        normed_conv0_outputs = torch.nn.functional.elu(normed_conv0_outputs)
        normed_conv0_outputs = torch.nn.functional.pad(normed_conv0_outputs, (1, 1,1, 1, 0, 0))
        conv1_outputs = self._conv1(normed_conv0_outputs)
        shortcut_modifier = self._shortcut(inputs)
        if self._downsample_stride > 1:
            shortcut_modifier = torch.nn.functional.avg_pool2d(shortcut_modifier, self._downsample_stride, self._downsample_stride)

        # Upsample the shortcut in the channel dimension if necessary
        if self._out_channels > self._in_channels:
            pad_dimension = (self._out_channels - self._in_channels) // 2
            shortcut_modifier = torch.nn.functional.pad(shortcut_modifier, [0,0,0,0,pad_dimension, pad_dimension])
            # NOTE: This code doesn't handle the case if _out_channels < _in_channels

        return conv1_outputs + shortcut_modifier


class BRegNextResidualBlock(torch.nn.Module):

    def __init__(self, n_blocks, in_channels, out_channels, downsample_stride=1):
        super(BRegNextResidualBlock, self).__init__()
        layers = [BReGNeXtResidualLayer(in_channels, out_channels, downsample_stride)] + [
            BReGNeXtResidualLayer(out_channels, out_channels, downsample_stride) for _ in range(n_blocks - 1)
        ]
        self._layer_stack = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self._layer_stack(inputs)


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

        self._conv0 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

        self.base_model = torch.nn.Sequential(
            # NOTE: The original BReGNeXt code uses a truncated normal initialization for this convolution, however
            # that is not implemented in PyTorch 1.7 - This defaults to a uniform initializer in PyTorch.

            BRegNextResidualBlock(n_blocks=1, in_channels=32, out_channels=32),
            CBAM(gate_channels=32, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            BRegNextResidualBlock(n_blocks=1, in_channels=32, out_channels=64, downsample_stride=2),
            CBAM(gate_channels=64, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            BRegNextResidualBlock(n_blocks=1, in_channels=64, out_channels=64),
            CBAM(gate_channels=64, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            BRegNextResidualBlock(n_blocks=1, in_channels=64, out_channels=128, downsample_stride=2),
            CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            BRegNextResidualBlock(n_blocks=1, in_channels=128, out_channels=128),
            CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            # torch.nn.AdaptiveAvgPool2d((1, 1)),  # it should be commented out if you use branches
        )

    def forward(self, x):
        base_feat = torch.nn.functional.pad(x, (1, 1, 1, 1, 0, 0))
        # print('padded_input_shape', base_feat.shape)  # 32 x 3 x 98 x 98
        base_feat = self._conv0(base_feat)
        base_out = self.base_model(base_feat)
        # print('base_out_shape', base_out.shape)  # 32 x 128 x 24 x 24

        return base_out


class ConvolutionalBranch(nn.Module):
    """
        Convolutional branches that compose the ensemble in ESRs. Each branch was trained on a sub-training
        set from the AffectNet dataset to learn complementary representations from the data (Siqueira et al., 2020).

        Note that, the second last layer provides eight discrete emotion labels whereas the last layer provides
        continuous values of arousal and valence levels.
    """

    def __init__(self):
        super(ConvolutionalBranch, self).__init__()

        self.branch_model = torch.nn.Sequential(
            BRegNextResidualBlock(n_blocks=1, in_channels=128, out_channels=128),
            CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            BRegNextResidualBlock(n_blocks=1, in_channels=128, out_channels=256, downsample_stride=2),
            CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            BRegNextResidualBlock(n_blocks=1, in_channels=256, out_channels=256),
            CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            BRegNextResidualBlock(n_blocks=1, in_channels=256, out_channels=512, downsample_stride=2),
            CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc_dimensional = nn.Linear(8, 2)
        self._fc0 = torch.nn.Linear(512, 8)  # it should be 512

    def forward(self, x_shared_representations):
        x_conv_branch = self.branch_model(x_shared_representations)
        x_conv_branch = x_conv_branch.view(-1, 512)
        # x_conv_branch = x_shared_representations.view(-1, 128)
        discrete_emotion = self._fc0(x_conv_branch)
        x_conv_branch = F.relu(discrete_emotion)
        continuous_affect = self.fc_dimensional(x_conv_branch)

        # Returns activations of the discrete emotion output layer and arousal and valence levels
        return discrete_emotion, continuous_affect

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
        x_to_last_conv_layer = self.pool(F.relu(self.bn2(self.conv2(x_to_last_conv_layer))))
        x_to_last_conv_layer = F.relu(self.bn3(self.conv3(x_to_last_conv_layer)))
        x_to_last_conv_layer = F.relu(self.bn4(self.conv4(x_to_last_conv_layer)))

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

        # Get shared representations
        x_shared_representations = self.base(x)

        # Add to the lists of predictions outputs from each convolutional branch in the ensemble
        for branch in self.convolutional_branches:
            output_emotion, output_affect = branch(x_shared_representations)
            emotions.append(output_emotion)
            affect_values.append(output_affect)

        return emotions, affect_values