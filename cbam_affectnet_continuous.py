#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiments on AffectNet for continuous emotion perception published at AAAI-20 (Siqueira et al., 2020).

Reference:
    Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
    Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
    (AAAI-20), pages 1–1, New York, USA.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# External Libraries
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import torch

# Standard Libraries
from os import path, makedirs
import copy

# Modules
from model.utils import udata, umath
from model.ml.esr_9_cbam import ESR


def evaluate(val_model_eval, val_loader_eval, val_criterion_eval, device_to_process, current_branch_on_training_val=0):
    cpu_device = torch.device('cpu')
    val_predictions = [[] for _ in range(val_model_eval.get_ensemble_size() + 1)]  # one list for each branch! so we will have 9 lists of predictions and 1 list for the ensemble result (so 10 lists in total)
    val_targets_valence = []
    val_targets_arousal = []

    for inputs_eval, labels_eval in val_loader_eval:
        inputs_eval, labels_eval = inputs_eval.to(device_to_process), labels_eval
        labels_eval_valence = labels_eval[:, 0].view(len(labels_eval[:, 0]), 1)
        labels_eval_arousal = labels_eval[:, 1].view(len(labels_eval[:, 1]), 1)

        emotions, affect_values, attn_heads = val_model_eval(inputs_eval)  # 9 lists of size n*2 each
        outputs_eval = affect_values[:val_model_eval.get_ensemble_size() - current_branch_on_training_val]  # only keep the output of branches which have been training

        # Ensemble prediction
        val_predictions_ensemble = torch.zeros(outputs_eval[0].size()).to(cpu_device)  # size is n*2 (size of one of the lists)

        for evaluate_branch in range(val_model_eval.get_ensemble_size() - current_branch_on_training_val):

            outputs_eval_cpu = outputs_eval[evaluate_branch].detach().to(cpu_device)

            val_predictions[evaluate_branch].extend(outputs_eval_cpu)
            val_predictions_ensemble += outputs_eval_cpu    # so the output of all the branches are summed up

        val_predictions[-1].extend(val_predictions_ensemble / (val_model_eval.get_ensemble_size() - current_branch_on_training_val))  # the mean of ensemble output is placed in the last position of val_prediction

        val_targets_valence.extend(labels_eval_valence)
        val_targets_arousal.extend(labels_eval_arousal)

    val_targets_valence = torch.stack(val_targets_valence)
    val_targets_arousal = torch.stack(val_targets_arousal)

    evaluate_val_losses = [[], []]   # valence, arousal (2 lists , each have 10 values, 1-9 for branch losses and 10 for ensemble loss)
    for evaluate_branch in range(val_model_eval.get_ensemble_size() + 1):
        if (evaluate_branch < (val_model_eval.get_ensemble_size() - current_branch_on_training_val)) or (evaluate_branch == val_model_eval.get_ensemble_size()):
            list_tensor = torch.stack(val_predictions[evaluate_branch])

            out_valence_eval = list_tensor[:, 0].view(len(list_tensor[:, 0]), 1)
            out_arousal_eval = list_tensor[:, 1].view(len(list_tensor[:, 1]), 1)

            evaluate_val_losses[0].append(torch.sqrt(val_criterion_eval(out_valence_eval, val_targets_valence)))
            evaluate_val_losses[1].append(torch.sqrt(val_criterion_eval(out_arousal_eval, val_targets_arousal)))
        else:
            evaluate_val_losses[0].append(torch.tensor(0))
            evaluate_val_losses[1].append(torch.tensor(0))

    return evaluate_val_losses


def plot(his_loss, his_val_loss_valence, his_val_loss_arousal, branch_idx, base_path_his):
    losses_plot = [[range(len(his_loss)), his_loss]]
    legends_plot_loss = ['Training']

    # Loss
    for b_plot in range(len(his_val_loss_valence)):
        losses_plot.append([range(len(his_val_loss_valence[b_plot])), his_val_loss_valence[b_plot]])
        legends_plot_loss.append('Validation ({}) (Val)'.format(b_plot + 1))

        losses_plot.append([range(len(his_val_loss_arousal[b_plot])), his_val_loss_arousal[b_plot]])
        legends_plot_loss.append('Validation ({}) (Aro)'.format(b_plot + 1))

    # Loss
    umath.plot(losses_plot,
               title='Training and Validation Losses vs. Epochs for Branch {}'.format(branch_idx),
               legends=legends_plot_loss,
               file_path=base_path_his,
               file_name='Loss_Branch_{}'.format(branch_idx),
               axis_x='Training Epoch',
               axis_y='Loss',
               limits_axis_y=(0.2, 0.6, 0.025))

    np.save(path.join(base_path_his, 'Loss_Branch_{}'.format(branch_idx)), np.array(his_loss))
    np.save(path.join(base_path_his, 'Loss_Val_Branch_{}_Valence'.format(branch_idx)), np.array(his_val_loss_valence))
    np.save(path.join(base_path_his, 'Loss_Val_Branch_{}_Arousal'.format(branch_idx)), np.array(his_val_loss_arousal))


def main():
    # Experimental variables
    base_path_experiment = "./experiments/AffectNet_Continuous/CBAM"
    name_experiment = "ESR_9_base_branch_AffectNet_Continuous_detdiv"
    base_path_to_dataset = "../FER_data/AffectNet"
    num_branches_trained_network = 9
    validation_interval = 1
    max_training_epoch = 2
    current_branch_on_training = 8  # it starts from branch 8 (last branch) and trains all the branches step by step

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

    # Load network trained on AffectNet
    net = ESR(device)
    ESR.load(device=device, ensemble_size=num_branches_trained_network)

    # fix the backbone
    for param in net.parameters():
        param.requires_grad = False
    for i in range(net.get_ensemble_size()):
        for p in net.convolutional_branches[i].fc_dimensional.parameters():
            p.requires_grad = True

    # Send params to device
    net.to_device(device)

    # Set optimizer
    optimizer = optim.SGD([{'params': net.base.parameters(), 'lr': 0.01, 'momentum': 0.9},
                           {'params': net.branches[0].parameters(), 'lr': 0.01, 'momentum': 0.9}])
    for b in range(1, net.get_ensemble_size()):
        optimizer.add_param_group({'params': net.branches[b].parameters(), 'lr': 0.001, 'momentum': 0.9})

    # Define criterion
    criterion = nn.MSELoss(reduction='mean')

    # Load validation set
    # max_loaded_images_per_label=100000 loads the whole validation set
    val_data = udata.AffectNetDimensional(idx_set=2,
                                          max_loaded_images_per_label=100000,
                                          transforms=None,  # why None? only train data is transformed? is it okay?
                                          is_norm_by_mean_std=False,    # why False?
                                          base_path_to_affectnet=base_path_to_dataset)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=8)

    # Fine-tune ESR-9
    for branch_on_training in range(num_branches_trained_network):
        # Load training data
        train_data = udata.AffectNetDimensional(idx_set=0,
                                                max_loaded_images_per_label=5000,
                                                transforms=transforms.Compose(data_transforms),
                                                is_norm_by_mean_std=False,
                                                base_path_to_affectnet=base_path_to_dataset)

        # Best network
        best_ensemble = net.to_state_dict()
        best_ensemble_rmse = 10000000.0

        # History
        history_loss = []
        history_val_loss_valence = [[] for _ in range(net.get_ensemble_size() + 1)]  # one list for each branch + 1 for ensemble (10 in total)
        history_val_loss_arousal = [[] for _ in range(net.get_ensemble_size() + 1)]  # one list for each branch + 1 for ensemble (10 in total)

        # Training branch
        for epoch in range(max_training_epoch):
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)

            running_loss = 0.0
            running_updates = 0

            batch = 0
            for inputs, labels in train_loader:
                batch += 1
                # Get the inputs
                inputs, labels = inputs.to(device), labels.to(device)
                labels_valence = labels[:, 0].view(len(labels[:, 0]), 1)
                labels_arousal = labels[:, 1].view(len(labels[:, 1]), 1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                emotions, affect_values, attn_heads = net(inputs)   # a list of 9 (ensemble_size) output arrays (9 * n*2), one array for each branch

                # Compute loss
                loss = 0.0
                for i_4 in range(net.get_ensemble_size() - current_branch_on_training):
                    out_valence = affect_values[i_4][:, 0].view(len(affect_values[i_4][:, 0]), 1)
                    out_arousal = affect_values[i_4][:, 1].view(len(affect_values[i_4][:, 1]), 1)

                    loss += torch.sqrt(criterion(out_valence, labels_valence))
                    loss += torch.sqrt(criterion(out_arousal, labels_arousal))

                # Backward
                loss.backward()

                # Optimize
                optimizer.step()

                # Save loss
                running_loss += loss.item()
                running_updates += 1

            # Statistics
            print('[Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f}'.format(
                net.get_ensemble_size() - current_branch_on_training, epoch + 1, max_training_epoch,
                running_loss / running_updates))

            # Validation
            if (batch % validation_interval) == 0:
                net.eval()

                val_loss = evaluate(net, val_loader, criterion, device, current_branch_on_training)

                # Add to history training and validation statistics
                history_loss.append(running_loss / running_updates)

                for b in range(net.get_ensemble_size()):
                    history_val_loss_valence[b].append(val_loss[0][b])  # we keep history loss for each branch which includes the loss values for all the epochs. val_loss keeps the loss for the current epoch and history_val_loss keeps the loss for all the epochs
                    history_val_loss_arousal[b].append(val_loss[1][b])  # so if we have for example 20 epochs in total, we have 20 values for each branch (20 for valence and 20 for arousal).

                # Add ensemble rmse to history
                history_val_loss_valence[-1].append(val_loss[0][-1])
                history_val_loss_arousal[-1].append(val_loss[1][-1])

                print('Validation - [Branch {:d}, Epochs {:d}--{:d}] Loss (V) - (A): ({}) - ({})'.format(
                    net.get_ensemble_size() - current_branch_on_training,
                    epoch + 1,
                    max_training_epoch,
                    [hvlv[-1] for hvlv in history_val_loss_valence],
                    [hvla[-1] for hvla in history_val_loss_arousal]
                ))  # for each branch, it prints the last value which is for the latest epoch! so it prints 10 values for valence and 10 for arousal! (for each branch it has num_epochs values in total)

                # Save best ensemble
                ensemble_rmse = float(history_val_loss_valence[-1][-1]) + float(history_val_loss_arousal[-1][-1])
                if ensemble_rmse <= best_ensemble_rmse:
                    best_ensemble_rmse = ensemble_rmse
                    best_ensemble = net.to_state_dict()

                    # Save network
                    ESR.save(best_ensemble, path.join(base_path_experiment, name_experiment, 'Saved Networks'),
                             current_branch_on_training)

                # Save graphs
                plot(history_loss, history_val_loss_valence, history_val_loss_arousal,
                     net.get_ensemble_size() - current_branch_on_training,
                     path.join(base_path_experiment, name_experiment))

                net.train()

        # Change branch on training
        if current_branch_on_training > 0:
            # Decrease max epoch
            max_training_epoch = 2  # I commented out

            # Reload best configuration
            net.reload(best_ensemble)

            # Send params to device
            net.to_device(device)

            # Set optimizer
            optimizer = optim.SGD([{'params': net.base.parameters(), 'lr': 0.001, 'momentum': 0.9},
                                   {'params': net.branches[
                                       net.get_ensemble_size() - current_branch_on_training].parameters(), 'lr': 0.01,
                                    'momentum': 0.9}])
            for b in range(net.get_ensemble_size()):
                if b != (net.get_ensemble_size() - current_branch_on_training):
                    optimizer.add_param_group({'params': net.branches[b].parameters(), 'lr': 0.001, 'momentum': 0.9})

            current_branch_on_training -= 1

        # Finish training after fine-tuning all branches
        else:
            break


if __name__ == "__main__":
    print("Processing...")
    main()
    print("Process has finished!")
