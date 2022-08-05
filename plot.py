
from model.utils import umath
import argparse
import numpy as np
from os import path


def plot(branch_idx, base_path_his):
    accuracies_plot = []
    legends_plot_acc = []
    losses_plot = []
    legends_plot_loss = []

    # Acc
    '''for b_plot in range(len(his_acc)):
        accuracies_plot.append([range(len(his_acc[b_plot])), his_acc[b_plot]])
        legends_plot_acc.append("Training ({})".format(b_plot + 1))

        accuracies_plot.append([range(len(his_val_acc[b_plot])), his_val_acc[b_plot]])
        legends_plot_acc.append("Validation ({})".format(b_plot + 1))'''

    # Ensemble acc
    for i in range(0, branch_idx):
        # Acc
        his_val_acc = np.load(path.join(base_path_his, "Acc_Val_Branch_{}.npy".format(i+1)))
        accuracies_plot.append([range(len(his_val_acc[-1])), his_val_acc[-1]])
        legends_plot_acc.append("Validation_Acc_E ({})".format(i+1))
        # Loss
        his_val_loss = np.load(path.join(base_path_his, "Loss_Val_Branch_{}.npy".format(i + 1)))
        losses_plot.append([range(len(his_val_loss[-1])), his_val_loss[-1]])
        legends_plot_loss.append("Validation_Loss_E ({})".format(i + 1))

    # Loss
    '''for b_plot in range(len(his_val_loss)):
        losses_plot.append([range(len(his_val_loss[b_plot])), his_val_loss[b_plot]])
        legends_plot_loss.append("Validation ({})".format(b_plot + 1))'''

    # Loss
    umath.plot(losses_plot,
               title="Validation Losses vs. Epochs for Branch {}".format(branch_idx),
               legends=legends_plot_loss,
               file_path=base_path_his,
               file_name="Loss_Branch_{}".format(branch_idx),
               axis_x="Training Epoch",
               axis_y="Loss")

    # Accuracy
    umath.plot(accuracies_plot,
               title="Validation Accuracies vs. Epochs for Branch {}".format(branch_idx),
               legends=legends_plot_acc,
               file_path=base_path_his,
               file_name="Acc_Branch_{}".format(branch_idx),
               axis_x="Training Epoch",
               axis_y="Accuracy",
               limits_axis_y=(0.0, 1.0, 0.025))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--acc_dir", default="./model/ml/trained_models/esr_9_cbam/results/Acc_Branch_9.npy")
    # parser.add_argument("--loss_dir", default="./model/ml/trained_models/esr_9_cbam/results/Loss_Branch_9.npy")
    # parser.add_argument("--val_acc_dir", default="./model/ml/trained_models/esr_9_cbam/results/Acc_Val_Branch_9.npy")
    # parser.add_argument("--val_loss_dir", default="./model/ml/trained_models/esr_9_cbam/results/Loss_Val_Branch_9.npy")
    parser.add_argument("--branch", default=9)
    parser.add_argument("--base_path_his", default='./model/ml/trained_models/esr_9/results')

    args = parser.parse_args()

    '''his_loss = np.load(args.loss_dir)
    his_acc = np.load(args.acc_dir)
    his_val_loss = np.load(args.val_loss_dir)
    his_val_acc = np.load(args.val_acc_dir)'''

    branch_idx = args.branch
    base_path_his = args.base_path_his


    plot(branch_idx, base_path_his)
