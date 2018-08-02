
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from deeplab.datasets import segmentation_dataset
from scipy.signal import savgol_filter
import copy
import os
import seaborn as sns
import cycler
import matplotlib as mpl
import operator
from matplotlib import patches


def get_results(events_path):
    step = []
    learning_rate = []
    total_loss = []
    for event in tf.train.summary_iterator(events_path):
        for v in event.summary.value:
            if v.tag == 'learning_rate':
                step.append(event.step)
                learning_rate.append(v.simple_value)
            if v.tag == 'total_loss_1':
                total_loss.append(v.simple_value)

    step = np.unique(step)

    return step, learning_rate, total_loss


if __name__=='__main__':

    set_fonts = 12
    num_plots = 4
    colormap = plt.cm.get_cmap('Dark2')
    color = colormap(np.linspace(0, 1, num_plots))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    events_path = './checkpoints_logs/train_logs/mobileNet/combined/binary_01/events.out.tfevents.1530423556.wr24'
    events_path = './checkpoints_logs/train_logs/mobileNet/combined/size_01/events.out.tfevents.1530423739.wr21'
    step_cr, lr_cr, tl_cr = get_results(events_path)
    events_path = './checkpoints_logs/train_logs/mobileNet/normal_lr/binary_01/events.out.tfevents.1532151003.wr26'
    events_path = './checkpoints_logs/train_logs/mobileNet/normal_lr/size_01/events.out.tfevents.1532151032.wr20'
    step_poly, lr_poly, tl_poly = get_results(events_path)

    mark = ['o', 'd']

    pl1, = plt.plot(step_cr, lr_cr)
    pl2, = plt.plot(step_poly, lr_poly)
    sc1 = plt.scatter(step_cr, lr_cr,
                      marker=mark[0], linewidths=0.2)
    sc2 = plt.scatter(step_poly, lr_poly,
                      marker=mark[1], linewidths=0.2)

    plt.tick_params(axis='both', which='major',
                    labelsize=set_fonts)
    plt.xlabel('Number of training steps',
               fontsize=set_fonts, labelpad=12)
    plt.ylabel('Learning rate', fontsize=set_fonts)
    plt.legend([(pl1, sc1), (pl2, sc2)], ['cosine restarts', 'poly'],
               bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=3,
               borderaxespad=0., prop={'size': set_fonts})
    plt.tight_layout()
    plt.grid(zorder=0, linestyle='dotted')
    plt.show()

    pl1, = plt.plot(step_cr, tl_cr)
    pl2, = plt.plot(step_poly, tl_poly)
    sc1 = plt.scatter(step_cr, tl_cr,
                      marker=mark[0], linewidths=0.2)
    sc2 = plt.scatter(step_poly, tl_poly,
                      marker=mark[1], linewidths=0.2)

    plt.tick_params(axis='both', which='major',
                    labelsize=set_fonts)
    plt.xlabel('Number of training steps',
               fontsize=set_fonts, labelpad=12)
    plt.ylabel('Total loss', fontsize=set_fonts)
    plt.legend([(pl1, sc1), (pl2, sc2)], ['cosine restarts', 'poly'],
               bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=3,
               borderaxespad=0., prop={'size': set_fonts})
    plt.tight_layout()
    plt.grid(zorder=0, linestyle='dotted')
    plt.show()
