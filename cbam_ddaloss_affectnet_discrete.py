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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# External Libraries
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
from os import path, makedirs
import argparse
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)

# Modules
from model.utils import udata, umath
from model.ml.esr_9_cbam_ddaloss import ESR


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



class DDA_Loss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512, batch_size=32):
        super(DDA_Loss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x_conv_branch = (x.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2)  # NxCxD (check dims)
        dist_center = -1 * (x_conv_branch.sum(2))
        scores = self.log_softmax(dist_center)
        ddaloss = self.nllloss(scores, labels) / self.batch_size / 2.0
        return ddaloss


class BranchDiversity(nn.Module):
    def __init__(self, ):
        super(BranchDiversity, self).__init__()
        self.direct_div = 0
        self.det_div = 0
        self.logdet_div = 0

    def forward(self, x):  # num_branch x batch_size x 6 x 6
        num_branches = x.size(0)
        gamma = 10
        snm = torch.zeros((num_branches, num_branches))

        # diversity between spatial attention heads
        for i in range(num_branches):
            for j in range(num_branches):
                if i != j:
                    diff = torch.exp(-1 * gamma * torch.sum(torch.square(x[i, :, :, :] - x[j, :, :, :]), (1, 2))) # batch_size
                    diff = torch.mean(diff)  # (1/num_branches) * torch.sum(diff)  # 1
                    snm[i, j] = diff
        self.direct_div = torch.sum(snm)
        self.det_div = -1 * torch.det(snm)
        self.logdet_div = -1 * torch.logdet(snm)

        return self

'''class CenterLoss(nn.Module):
    def __init__(self, ):
        super(CenterLoss, self).__init__()
        self.totloss = 0
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

        def forward(self, dist_center, label, batch_size):
            for i in range(len(dist_center)):
                dist = dist_center[i]
                scores = self.log_softmax(dist)
                ddaloss = self.nllloss(scores, label) / batch_size / 2.0
                self.totloss += ddaloss
            return self.totloss'''


def main(args):
    # Experimental variables
    max_training_epoch = args.max_training_epoch

    # Make dir
    if not path.isdir(path.join(args.base_path_experiment, args.name_experiment)):
        makedirs(path.join(args.base_path_experiment, args.name_experiment))

    # Define transforms
    data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomAffine(degrees=30, translate=(.1, .1), scale=(1.0, 1.25),
                                               resample=Image.BILINEAR)]

    # Running device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Define criterion
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dda = DDA_Loss(device)
    criterion_div = BranchDiversity()

    print("Starting: {}".format(str(args.name_experiment)))
    print("Running on {}".format(device))

    # Initialize network
    net = ESR(device)

    # Add first branch
    net.add_branch()

    # Send to running device
    net.to_device(device)

    '''dtype = torch.cuda.FloatTensor
    centers = Variable(torch.randn(512, 8).type(dtype), requires_grad=False)
    centers = centers.to(device)
    # centers = nn.Parameter(torch.FloatTensor(512, 8))'''

    # Set optimizer
    optimizer = optim.SGD([{'params': net.base.parameters(), 'lr': 0.1, 'momentum': 0.9},
                           {'params': net.convolutional_branches[-1].parameters(), 'lr': 0.1, 'momentum': 0.9},
                           {'params': criterion_dda.parameters(), 'lr': 0.1, 'momentum': 0.9}])

    # Load validation set. max_loaded_images_per_label=100000 loads the whole validation set
    val_data = udata.AffectNetCategorical(idx_set=2,
                                          max_loaded_images_per_label=100000,
                                          transforms=None,
                                          is_norm_by_mean_std=False,
                                          base_path_to_affectnet=args.base_path_to_dataset)

    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=8)

    # Train ESR-9
    for branch_on_training in range(args.num_branches_trained_network):
        # Load training data
        train_data = udata.AffectNetCategorical(idx_set=0,
                                                max_loaded_images_per_label=5000,
                                                transforms=transforms.Compose(data_transforms),
                                                is_norm_by_mean_std=False,
                                                base_path_to_affectnet=args.base_path_to_dataset)

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

            for inputs, labels in train_loader:
                # Get the inputs
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                emotions, x_conv, attn_heads = net(inputs)
                confs_preds = [torch.max(o, 1) for o in emotions]

                # Compute loss
                loss = 0.0
                for i_4 in range(net.get_ensemble_size()):
                    preds = confs_preds[i_4][1]
                    running_corrects[i_4] += torch.sum(preds == labels).cpu().numpy()
                    # CE loss
                    loss += criterion_ce(emotions[i_4], labels)
                    # DDA loss
                    loss += criterion_dda(x_conv[i_4], labels)

                # branch_Div
                # loss += criterion_div(attn_heads).det_div

                # Backward
                loss.backward(retain_graph=True)

                # Optimize
                optimizer.step()

                # Save loss
                running_loss += loss.item()
                running_updates += 1

            scheduler.step()
            # Statistics
            print('[Branch {:d}, '
                  'Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}'.format(net.get_ensemble_size(),
                                                                   epoch + 1,
                                                                   max_training_epoch,
                                                                   running_loss / running_updates,
                                                                   np.array(running_corrects) / len(train_data)))
            # Validation
            if ((epoch % args.validation_interval) == 0) or ((epoch + 1) == max_training_epoch):
                net.eval()
                val_loss, val_corrects = evaluate(net, val_loader, criterion_ce, device)
                print('Validation - [Branch {:d}, '
                      'Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}'.format(net.get_ensemble_size(),
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
                    ESR.save(best_ensemble, path.join(args.base_path_experiment, args.name_experiment, 'SavedNetworks'),
                             net.get_ensemble_size())

                # Save graphs
                plot(history_loss, history_acc, history_val_loss, history_val_acc,
                     net.get_ensemble_size(), path.join(args.base_path_experiment, args.name_experiment))

                # Set network to training mode
                net.train()

        # Change branch on training
        if net.get_ensemble_size() < args.num_branches_trained_network:
            # Decrease maximum training epoch
            max_training_epoch = args.max_finetune_epoch

            # Reload best configuration
            net.reload(best_ensemble)

            # Add branch
            net.add_branch()

            # Send params to device
            net.to_device(device)

            # Set optimizer for base and the new branch
            optimizer = optim.SGD([{'params': net.base.parameters(), 'lr': 0.01, 'momentum': 0.9},
                                   {'params': net.convolutional_branches[-1].parameters(), 'lr': 0.1,
                                    'momentum': 0.9},
                                   {'params': criterion_dda.parameters(), 'lr': 0.1, 'momentum': 0.9}])

            # Set optimizer for the trained branches
            if not args.freeze_trained_branches:
                for b in range(net.get_ensemble_size() - 1):
                    optimizer.add_param_group({'params': net.convolutional_branches[b].parameters(), 'lr': 0.01,
                                               'momentum': 0.9})

        # Finish training after training all branches
        else:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path_experiment", default="./experiments/AffectNet_Discrete/DDALoss")
    parser.add_argument("--name_experiment", default="CBAM_ESR_9_bb_ddaloss")
    parser.add_argument("--base_path_to_dataset", default="../FER_data/AffectNet/")
    parser.add_argument("--num_branches_trained_network", default=15)
    parser.add_argument("--validation_interval", default=1)
    parser.add_argument("--max_training_epoch", default=50)
    parser.add_argument("--max_finetune_epoch", default=20)
    parser.add_argument("--freeze_trained_branches", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu_indx", default="1")

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_indx

    print("Processing...")
    main(args)
    print("Process has finished!")


