#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiments on AffectNet for discrete emotion perception published at AAAI-20 (Siqueira et al., 2020).

Reference:
    Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
    Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
    (AAAI-20), pages 1–1, New York, USA.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"


# External Libraries
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
from os import path, makedirs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Modules
from model.utils import udata, umath
from model.ml.esr_5 import ESR


def evaluate(val_model_eval, val_loader_eval, val_criterion_eval, device_to_process="cpu", current_branch_on_training_val=0):
    running_val_loss = [0.0 for _ in range(val_model_eval.get_ensemble_size())]
    running_val_corrects = [0 for _ in range(val_model_eval.get_ensemble_size() + 1)]
    running_val_steps = [0 for _ in range(val_model_eval.get_ensemble_size())]

    for inputs_eval, labels_eval in val_loader_eval:
        inputs_eval, labels_eval = inputs_eval.to(device_to_process), labels_eval.to(device_to_process)
        outputs_eval, _, _ = val_model_eval(inputs_eval)
        outputs_eval = outputs_eval[:val_model_eval.get_ensemble_size() - current_branch_on_training_val]   # a list of #n torch tensors #n denotes the number of branches

        # Ensemble prediction
        overall_preds = torch.zeros(outputs_eval[0].size()).to(device_to_process)  # size: batchsize * 8
        for o_eval, outputs_per_branch_eval in enumerate(outputs_eval, 0):  # go through the list and take the output of each branch which is of size batchsize*8
            _, preds_eval = torch.max(outputs_per_branch_eval, 1)  # preds_eval is the indices of max value for each sample so it is an array of size torch.size([batchsize])

            running_val_corrects[o_eval] += torch.sum(preds_eval == labels_eval).cpu().numpy()  # o_eval is the branch index
            loss_eval = val_criterion_eval(outputs_per_branch_eval, labels_eval)
            running_val_loss[o_eval] += loss_eval.item()
            running_val_steps[o_eval] += 1

            for v_i, v_p in enumerate(preds_eval, 0):
                overall_preds[v_i, v_p] += 1

        # Compute accuracy of ensemble predictions
        _, preds_eval = torch.max(overall_preds, 1)
        running_val_corrects[-1] += torch.sum(preds_eval == labels_eval).cpu().numpy()

    for b_eval in range(val_model_eval.get_ensemble_size()):
        div = running_val_steps[b_eval] if running_val_steps[b_eval] != 0 else 1
        running_val_loss[b_eval] /= div

    return running_val_loss, running_val_corrects


def plot(his_loss, his_acc, his_val_loss, his_val_acc, branch_idx, base_path_his):
    accuracies_plot = []
    legends_plot_acc = []
    losses_plot = [[range(len(his_loss)), his_loss]]
    legends_plot_loss = ["Training"]

    # Acc
    for b_plot in range(len(his_acc)):
        accuracies_plot.append([range(len(his_acc[b_plot])), his_acc[b_plot]])
        legends_plot_acc.append("Training ({})".format(b_plot + 1))

        accuracies_plot.append([range(len(his_val_acc[b_plot])), his_val_acc[b_plot]])
        legends_plot_acc.append("Validation ({})".format(b_plot + 1))

    # Ensemble acc
    accuracies_plot.append([range(len(his_val_acc[-1])), his_val_acc[-1]])
    legends_plot_acc.append("Validation (E)")

    # Loss
    for b_plot in range(len(his_val_loss)):
        losses_plot.append([range(len(his_val_loss[b_plot])), his_val_loss[b_plot]])
        legends_plot_loss.append("Validation ({})".format(b_plot + 1))

    # Loss
    umath.plot(losses_plot,
               title="Training and Validation Losses vs. Epochs for Branch {}".format(branch_idx),
               legends=legends_plot_loss,
               file_path=base_path_his,
               file_name="Loss_Branch_{}".format(branch_idx),
               axis_x="Training Epoch",
               axis_y="Loss")

    # Accuracy
    umath.plot(accuracies_plot,
               title="Training and Validation Accuracies vs. Epochs for Branch {}".format(branch_idx),
               legends=legends_plot_acc,
               file_path=base_path_his,
               file_name="Acc_Branch_{}".format(branch_idx),
               axis_x="Training Epoch",
               axis_y="Accuracy",
               limits_axis_y=(0.0, 1.0, 0.025))

    # Save plots
    np.save(path.join(base_path_his, "Loss_Branch_{}".format(branch_idx)), np.array(his_loss))
    np.save(path.join(base_path_his, "Acc_Branch_{}".format(branch_idx)), np.array(his_acc))
    np.save(path.join(base_path_his, "Loss_Val_Branch_{}".format(branch_idx)), np.array(his_val_loss))
    np.save(path.join(base_path_his, "Acc_Val_Branch_{}".format(branch_idx)), np.array(his_val_acc))


class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()

    def forward(self, x):
        num_head = x.size(1)
        # print('num_head size', x.size())  # batch_size * num_branches * 512

        if num_head > 1:
            var = x.var(dim=1).mean()
            # print('var_shape', var.shape)
            loss = torch.log(1 + num_head / (var + 1e-10))
        else:
            loss = 0
        return loss


class FeatureDiversity(nn.Module):
    def __init__(self, ):
        super(FeatureDiversity, self).__init__()

    def forward(self, x):
        num_features = x.size(2)
        diff = 0
        for i in range(num_features):
            for j in range(num_features):
                diff += torch.square(x[:, :, i] - x[:, :, j])
        diff = 1/(2*num_features*(num_features-1)) * diff
        diff = torch.sum(diff, 1)
        div = diff.mean()
        return div


def main():
    # Experimental variables
    base_path_experiment = "./experiments/AffectNet_Discrete/"
    name_experiment = "ESR_9-AffectNet_Discrete"
    base_path_to_dataset = "../FER_data/AffectNet/"
    num_branches_trained_network = 9
    validation_interval = 1
    max_training_epoch = 50

    # Make dir
    if not path.isdir(path.join(base_path_experiment, name_experiment)):
        makedirs(path.join(base_path_experiment, name_experiment))

    # Define transforms
    data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomAffine(degrees=30,
                                               translate=(.1, .1),
                                               scale=(1.0, 1.25),
                                               resample=Image.BILINEAR)]

    # Running device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting: {}".format(str(name_experiment)))
    print("Running on {}".format(device))

    # Initialize network
    net = ESR(device)

    # Add first branch
    net.add_branch()

    # Send to running device
    net.to_device(device)

    # Set optimizer
    optimizer = optim.SGD([{'params': net.base.parameters(), 'lr': 0.1, 'momentum': 0.9},
                           {'params': net.convolutional_branches[-1].parameters(), 'lr': 0.1, 'momentum': 0.9}])

    # Define criterion
    criterion = nn.CrossEntropyLoss()
    attn_criterion = PartitionLoss()
    diversity = FeatureDiversity()

    # Load validation set
    # max_loaded_images_per_label=100000 loads the whole validation set
    val_data = udata.AffectNetCategorical(idx_set=2,
                                          max_loaded_images_per_label=100000,
                                          transforms=None,
                                          is_norm_by_mean_std=False,
                                          base_path_to_affectnet=base_path_to_dataset)

    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=8)

    # Train ESR-9
    for branch_on_training in range(num_branches_trained_network):
        # Load training data
        train_data = udata.AffectNetCategorical(idx_set=0,
                                                max_loaded_images_per_label=5000,
                                                transforms=transforms.Compose(data_transforms),
                                                is_norm_by_mean_std=False,
                                                base_path_to_affectnet=base_path_to_dataset)

        # Best network
        best_ensemble = net.to_state_dict()
        best_ensemble_acc = 0.0

        # Initialize scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)

        # History
        history_loss = []
        history_acc = [[] for _ in range(net.get_ensemble_size())]
        history_val_loss = [[] for _ in range(net.get_ensemble_size())]
        history_val_acc = [[] for _ in range(net.get_ensemble_size() + 1)]

        # Training branch
        for epoch in range(max_training_epoch):
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)

            running_loss = 0.0
            running_corrects = [0.0 for _ in range(net.get_ensemble_size())]
            running_updates = 0

            scheduler.step()

            for inputs, labels in train_loader:
                # Get the inputs
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                emotions, affect_values, heads = net(inputs)
                confs_preds = [torch.max(o, 1) for o in emotions]

                # Compute loss
                loss = 0.0
                for i_4 in range(net.get_ensemble_size()):
                    preds = confs_preds[i_4][1]
                    running_corrects[i_4] += torch.sum(preds == labels).cpu().numpy()
                    loss += criterion(emotions[i_4], labels)

                loss += attn_criterion(heads)    # partition loss between different attention heads (maximize the difference between them)
                print('atten_loss', attn_criterion(heads) )
                # div = diversity(heads)  # diversity between different channels of attention
                # print('loss', div)

                # Backward
                loss.backward()

                # Optimize
                optimizer.step()

                # Save loss
                running_loss += loss.item()
                running_updates += 1

            # Statistics
            print('[Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}'.format(net.get_ensemble_size(),
                                                                                 epoch + 1,
                                                                                 max_training_epoch,
                                                                                 running_loss / running_updates,
                                                                                 np.array(running_corrects) / len(train_data)))
            # Validation
            if ((epoch % validation_interval) == 0) or ((epoch + 1) == max_training_epoch):
                net.eval()

                val_loss, val_corrects = evaluate(net, val_loader, criterion, device)

                print('Validation - [Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}'.format(net.get_ensemble_size(),
                                                                                                  epoch + 1,
                                                                                                  max_training_epoch,
                                                                                                  val_loss[-1],
                                                                                                  np.array(val_corrects) / len(val_data)))

                # Add to history training and validation statistics
                history_loss.append(running_loss / running_updates)

                for i_4 in range(net.get_ensemble_size()):
                    history_acc[i_4].append(running_corrects[i_4] / len(train_data))

                for b in range(net.get_ensemble_size()):
                    history_val_loss[b].append(val_loss[b])
                    history_val_acc[b].append(float(val_corrects[b]) / len(val_data))

                # Add ensemble accuracy to history
                history_val_acc[-1].append(float(val_corrects[-1]) / len(val_data))

                # Save best ensemble
                ensemble_acc = (float(val_corrects[-1]) / len(val_data))
                if ensemble_acc >= best_ensemble_acc:
                    best_ensemble_acc = ensemble_acc
                    best_ensemble = net.to_state_dict()

                    # Save network
                    ESR.save(best_ensemble, path.join(base_path_experiment, name_experiment, 'Saved Networks'),
                                  net.get_ensemble_size())

                # Save graphs
                plot(history_loss, history_acc, history_val_loss, history_val_acc,
                     net.get_ensemble_size(), path.join(base_path_experiment, name_experiment))

                # Set network to training mode
                net.train()

        # Change branch on training
        if net.get_ensemble_size() < num_branches_trained_network:
            # Decrease maximum training epoch
            max_training_epoch = 20

            # Reload best configuration
            net.reload(best_ensemble)

            # Add branch
            net.add_branch()

            # Send params to device
            net.to_device(device)

            # Set optimizer
            optimizer = optim.SGD([{'params': net.base.parameters(), 'lr': 0.01, 'momentum': 0.9},
                                   {'params': net.convolutional_branches[-1].parameters(), 'lr': 0.1, 'momentum': 0.9}])
            for b in range(net.get_ensemble_size() - 1):
                optimizer.add_param_group({'params': net.convolutional_branches[b].parameters(), 'lr': 0.01, 'momentum': 0.9})

        # Finish training after training all branches
        else:
            break


if __name__ == "__main__":
    print("Processing...")
    main()
    print("Process has finished!")
