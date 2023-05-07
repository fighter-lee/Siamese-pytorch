import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

if __name__ == "__main__":
    if False:
        current_file = os.path.abspath(__file__)
        root = os.path.dirname(os.path.dirname(current_file))
        loss_txt = os.path.join(root, "rellogs", "resnet", "epoch_loss.txt")
        val_loss_txt = os.path.join(root, "rellogs", "resnet", "epoch_val_loss.txt")
        epoch_loss_png = os.path.join(root, "rellogs", "resnet", "epoch_loss.png")
        losses = []
        with open(loss_txt, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline().strip()
                if not line == '':
                    losses.append(float(line))

        val_losses = []
        with open(val_loss_txt, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline().strip()
                if not line == '':
                    val_losses.append(float(line))
        iters = range(len(losses))

        plt.figure()
        plt.plot(iters, losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, val_losses, 'coral', linewidth=2, label='val loss')
        try:
            if len(losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(val_losses, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(epoch_loss_png)

        plt.cla()
        plt.close("all")
    else:
        current_file = os.path.abspath(__file__)
        root = os.path.dirname(os.path.dirname(current_file))
        loss_txt = os.path.join(root, "rellogs", "vgg", "epoch_vgg_loss.txt")
        val_loss_txt = os.path.join(root, "rellogs", "vgg", "epoch_resnet_loss.txt")
        epoch_loss_png = os.path.join(root, "rellogs", "vgg", "epoch_loss.png")
        losses = []
        with open(loss_txt, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline().strip()
                if not line == '':
                    losses.append(float(line))

        val_losses = []
        with open(val_loss_txt, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline().strip()
                if not line == '':
                    val_losses.append(float(line))
        iters = range(len(losses))

        plt.figure()
        # plt.plot(iters, losses, 'red', linewidth=2, label='train loss')
        # plt.plot(iters, val_losses, 'coral', linewidth=2, label='val loss')
        try:
            if len(losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='VGG16 val loss')
            plt.plot(iters, scipy.signal.savgol_filter(val_losses, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='ResNet50 val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(epoch_loss_png)

        plt.cla()
        plt.close("all")