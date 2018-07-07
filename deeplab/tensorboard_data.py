
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

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('dataset', 'atWork_binary',
                          'Name of the segmentation dataset.')

flags.DEFINE_multi_string('events_dir', None,
                          'Directory of the events file.')

flags.DEFINE_integer('final_step', 30000,
                     'Final step of logs in the event file.')

flags.DEFINE_string('metric_to_plot', 'confusion_matrix',
                    'can be miou, class_iou, confusion_matrix')

colormap = np.asarray([[75, 25, 230], [75, 180, 60],
                       [25, 225, 255], [200, 130, 0], [48, 130, 245],
                       [180, 30, 145], [240, 240, 70], [230, 50, 240],
                       [60, 245, 210], [128, 128, 0], [255, 190, 230],
                       [40, 110, 170], [0, 0, 128], [195, 255, 170],
                       [0, 128, 128], [180, 215, 255], [128, 0, 0],
                       [0, 0, 0], [128, 128, 128]], dtype=np.uint8)


def get_results(events_path, class_names, cls_to_iou):

    step = []
    miou = []
    cls_to_iou = copy.deepcopy(cls_to_iou)
    confusion_rc_len = len(class_names) + 1

    confusion_string = None
    confusion_matrix = None
    for event in tf.train.summary_iterator(events_path):
        step.append(event.step)
        for v in event.summary.value:
            if np.any(class_names == v.tag):
                cls_to_iou[v.tag].append(v.simple_value * 100.)

            if event.step == FLAGS.final_step and v.tag == 'confusion_matrix':
                confusion_string = np.array(v.tensor.string_val)
                confusion_string = np.reshape(confusion_string,
                                              (confusion_rc_len,
                                               confusion_rc_len))
                confusion_matrix = confusion_string[:, 1:confusion_rc_len][
                                   1:confusion_rc_len, :]
                confusion_matrix = np.array(confusion_matrix, dtype=np.float64)

            if v.tag == 'miou_1.0':
                miou.append(v.simple_value * 100.)

    step = np.unique(step)

    return step, miou, cls_to_iou, confusion_matrix, confusion_string


def plot_class_ious(step, cls_to_iou, set_fonts, label_def,
                    cls_to_percentage):

    plot_objs = []

    cls_to_iou = {key.split('/')[-1]: value[-1]
                  for key, value in cls_to_iou.items()}

    cls_to_iou.pop('background', None)
    cls_to_percentage.pop('background', None)

    max_class_iou = np.sum(cls_to_iou.values())
    cls_to_iou = {key: value/max_class_iou
                 for key, value in cls_to_iou.items()}

    max_percentage = np.sum(cls_to_percentage.values())
    cls_to_percentage = {key: value/max_percentage
                      for key, value in cls_to_percentage.items()}

    sorted_classes = np.array(sorted(cls_to_percentage.items(),
                            key=operator.itemgetter(1)))[:,0]

    iou_vals = [cls_to_iou[cls_iou] for
                cls_iou in sorted_classes]
    percentage_vals = [cls_to_percentage[cls_iou] for
                       cls_iou in sorted_classes]

    x = np.arange(0, len(sorted_classes))
    plt.bar(x + 0.3, iou_vals, width=0.3, zorder=3, label='IOU')
    plt.bar(x, percentage_vals, width=0.3,
            align='center', zorder=3, label='Percentage')
    plt.xticks(x, sorted_classes, rotation=80)
    plt.grid(zorder=0, axis='y')

    # plt.bar(cls_to_iou.keys(), cls_to_iou.values())
    # plt.bar(cls_to_percentage.keys(), cls_to_percentage.values())

    # for index, (key, values) in enumerate(cls_to_iou.items()):
    #     values = savgol_filter(values, 7, 3)
    #     pl, = plt.plot(step, values, label=key,
    #                   c=np.flip(colormap[index], 0)/255.)
    #     plot_objs.append(pl)

    plt.tick_params(axis='both', which='major', labelsize=set_fonts)
    # plt.xlabel('Number of training steps', fontsize=set_fonts, labelpad=12)
    plt.ylabel('Normalized values', fontsize=set_fonts)
    #plt.legend(plot_objs, label_def.values(), fontsize=set_fonts, loc=4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mious(miou_list, step_list, labels_list, set_fonts,
               model_variant='mobileNet'):

    mark = ['o', 'd', '|', 's', 's', '|', 'd', 'o']
    plot_objs = []

    for index, (miou, step) in enumerate(zip(miou_list, step_list)):
        miou = savgol_filter(miou, 5, 2)
        pl, = plt.plot(step, miou)
        miou_new, step_new = miou, step
        if (model_variant == 'xception' or
                'xception' in labels_list[index]):
            miou_new = miou[0::2]
            step_new = step[0::2]
        sc = plt.scatter(step_new, miou_new,
                         marker=mark[index], linewidths=0.2)
        plot_objs.append((pl, sc))
        print(labels_list[index], '=', round(miou[-1], 2))

    plt.tick_params(axis='both', which='major',
                    labelsize=set_fonts)
    plt.xlabel('Number of training steps',
               fontsize=set_fonts, labelpad=12)
    plt.ylabel('Mean IOU (%)', fontsize=set_fonts)
    plt.legend(plot_objs, labels_list,
               fontsize=set_fonts, loc=4)
    plt.tight_layout()
    plt.show()


def plot_confusion(confusion_matrix, label_def):

    fig, ax = plt.subplots()
    correct_predictions = np.diag(confusion_matrix)
    second_max = sorted(correct_predictions)[-2]
    confusion_matrix = confusion_matrix/(
        np.sum(confusion_matrix, axis=1)[np.newaxis].T)
    confusion_matrix = confusion_matrix * 100
    confusion_matrix = np.round(confusion_matrix, 2)

    sns.heatmap(confusion_matrix, xticklabels=label_def.values(),
                yticklabels=label_def.values(), vmin=0, vmax=50,
                cmap=sns.color_palette('Blues', n_colors=30),
                annot=True, annot_kws={"size": 12})
    ax.xaxis.tick_top()
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.show()


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    set_fonts = 12
    num_plots = 4

    if len(FLAGS.dataset) != len(FLAGS.events_dir):
        raise ValueError('Number of datasets {} is not equal to'
                         'number of events {}.'.format(FLAGS.dataset,
                                                       FLAGS.events_dir))

    colormap = plt.cm.get_cmap('Dark2')
    color = colormap(np.linspace(0, 1, num_plots))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    miou_list = []
    step_list = []

    for index, (dataset, event_dir) in enumerate(
            zip(FLAGS.dataset, FLAGS.events_dir)):

        event_file = os.path.join(event_dir, os.listdir(
            event_dir)[0])
        label_def = segmentation_dataset.get_label_def(dataset)
        class_names = np.array(label_def.values())
        class_names = np.array(['class/' + cls for cls in class_names])
        cls_to_iou = {key: [] for key in class_names}

        (step, miou, cls_to_iou,
         confusion_matrix, _) = get_results(event_file, class_names, cls_to_iou)
        miou_list.append(miou)
        step_list.append(step)

        if FLAGS.metric_to_plot == 'class_iou':
            plot_class_ious(step, cls_to_iou, set_fonts, label_def,
                            segmentation_dataset.get_cls_to_percentage(
                                dataset))

        elif FLAGS.metric_to_plot == 'confusion_matrix':
            plot_confusion(confusion_matrix, label_def)

    if FLAGS.metric_to_plot == 'miou':
        # plot_mious(miou_list, step_list, FLAGS.dataset, set_fonts,
        #            model_variant='xception')

        miou_list = np.array(miou_list)
        step_list = np.array(step_list)
        #labels_list = ['VB: mobileNet', 'WB: mobileNet', 'VB: xception']
        #labels_list = ['All training data', 'Real training data', 'Artificial Training data']
        #labels_list = ['Real training data', 'Artificial Training data']
        #labels_list = ['PASCAL pretrained', 'Binary pretrained']
        labels_list = ['VB: PASCAL pretrained', 'VB: Binary pretrained',
                       'WB: PASCAL pretrained', 'WB: Binary pretrained']
        select_idx = [idx for idx, dataset in enumerate(FLAGS.dataset)
                      if 'size' in dataset]
        plot_mious(miou_list[select_idx], step_list[select_idx],
                   labels_list, set_fonts)


if __name__ == '__main__':
    flags.mark_flag_as_required('events_dir')
    tf.app.run()
