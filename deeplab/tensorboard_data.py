
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

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('dataset', 'atWork_binary',
                          'Name of the segmentation dataset.')

flags.DEFINE_multi_string('events_dir', None,
                          'Directory of the events file.')

flags.DEFINE_integer('final_step', 30000,
                     'Final step of logs in the event file.')

flags.DEFINE_string('metric_to_plot', 'miou',
                    'can be miou, class_iou, confusion_matrix, quant_results')

flags.DEFINE_bool('all_variants', True,
                  'Plot all variants together or not.')

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


def plot_class_ious_infer(iou_vals, percent_vals, set_fonts, combine):

    x = np.arange(0, len(iou_vals) / combine)
    iou_vals = np.reshape(iou_vals,
                          (len(iou_vals) / combine, combine))
    iou_vals = np.sum(iou_vals, axis=1)
    percent_vals = np.reshape(percent_vals,
                              (len(percent_vals) / combine, combine))
    percent_vals = np.sum(percent_vals, axis=1)

    plt.bar(x + 0.3, iou_vals, width=0.3, zorder=3, label='IOU')
    plt.bar(x, percent_vals, width=0.3,
            align='center', zorder=3, label='Percentage of pixels')
    plt.xticks([])
    plt.grid(zorder=0, axis='y')

    plt.tick_params(axis='both', which='major', labelsize=set_fonts)
    plt.xlabel('Combine every 2 classes', fontsize=set_fonts)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=3,
               borderaxespad=0., prop={'size': set_fonts})


def plot_class_ious(cls_to_iou, set_fonts,
                    cls_to_percentage):

    cls_to_iou = {key.split('/')[-1]: np.mean(value)
                  for key, value in cls_to_iou.items()}

    cls_to_iou.pop('background', None)
    cls_to_percentage.pop('background', None)

    sum_class_iou = np.sum(cls_to_iou.values())
    cls_to_iou = {key: value/sum_class_iou
                  for key, value in cls_to_iou.items()}

    sum_percentage = np.sum(cls_to_percentage.values())
    cls_to_percentage = {key: value/sum_percentage
                         for key, value in cls_to_percentage.items()}

    sorted_classes = np.array(sorted(cls_to_percentage.items(),
                              key=operator.itemgetter(1)))[:,0]

    iou_vals = [cls_to_iou[cls_iou] for
                cls_iou in sorted_classes]
    percentage_vals = [cls_to_percentage[cls_iou] for
                       cls_iou in sorted_classes]

    x = np.arange(0, len(sorted_classes))

    figure = plt.figure(figsize=(7, 6))
    figure.add_subplot(2, 1, 1)
    plot_class_ious_infer(iou_vals, percentage_vals, set_fonts, 2)

    figure.add_subplot(2, 1, 2)
    plt.bar(x + 0.3, iou_vals, width=0.3, zorder=3, label='IOU')
    plt.bar(x, percentage_vals, width=0.3,
            align='center', zorder=3, label='Percentage of pixels')
    plt.xticks(x, sorted_classes, rotation=80)
    plt.grid(zorder=0, axis='y')

    plt.tick_params(axis='both', which='major', labelsize=set_fonts)
    plt.ylabel('Normalized values', fontsize=set_fonts)
    plt.tight_layout()
    plt.show()


def plot_miou_bars(miou, set_fonts, labels_list):

    plot_objs = []
    collect_mious = []
    for index, m in enumerate(miou):
        collect_mious.append(m[-1])

    collect_mious = np.reshape(collect_mious, (8, 4))

    mob_miou = collect_mious[0:4, :]
    mob_miou = np.array(mob_miou)[np.newaxis].T
    xcep_miou = collect_mious[4:8, :]
    xcep_miou = np.array(xcep_miou)[np.newaxis].T
    mob_xvals = [0, 4, 8, 12]
    xcep_xvals = [2, 6, 10, 14]
    offset = [-0.3, 0, 0.3, 0.6]

    x_vals = np.arange(0, 15, 2)
    x_ticks = ['full\nMobileNetv2', 'full\nXception', 'size\nMobileNetv2',
               'size\nXception', 'shape\nMobileNetv2', 'shape\nXception',
               'binary\nMobileNetv2', 'binary\nXception']

    _ = plt.figure(figsize=(8, 4))
    plt.ylim([30, 100])

    for index, m in enumerate(mob_miou):
        mob_objs = []
        for i, val in enumerate(m):
            mob_objs.append(plt.bar(mob_xvals[index] + offset[i], val,
                                    width=0.3, zorder=3)[0])
        plot_objs.append(mob_objs)

    for index, m in enumerate(xcep_miou):
        xcep_objs = []
        for i, val in enumerate(m):
            xcep_objs.append(plt.bar(xcep_xvals[index] + offset[i], val,
                                     width=0.3, zorder=3)[0])
        plot_objs.append(xcep_objs)

    plot_objs = np.array(plot_objs)[np.newaxis].T

    plot_objs = plot_objs.reshape(4, 8)
    new_plots = []

    for obj in plot_objs:
        new_plots.append(tuple(obj))

    plt.grid(zorder=0, axis='y')
    plt.xticks(x_vals, x_ticks, rotation=80)
    plt.tick_params(axis='both', which='major', labelsize=set_fonts)
    plt.ylabel('mIOU (%)', fontsize=set_fonts)
    plt.legend(new_plots, labels_list, bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=2,
               borderaxespad=0., prop={'size': set_fonts})
    plt.tight_layout()
    plt.show()


def plot_class_balance(cls_iou, set_fonts):

    cls_iou_act = {key.split('/')[-1]: value[-1]
                   for key, value in cls_iou[1].items()}
    cls_iou_bal = {key.split('/')[-1]: value[-1]
                   for key, value in cls_iou[0].items()}

    sorted_classes = np.array(sorted(cls_iou_act.items(),
                              key=operator.itemgetter(1)))[:,0]

    x = np.arange(0, len(sorted_classes))

    iou_vals_act = [cls_iou_act[c_i] for
                    c_i in sorted_classes]
    iou_vals_bal = [cls_iou_bal[c_i] for
                    c_i in sorted_classes]

    _ = plt.figure(figsize=(8, 4))
    plt.bar(x + 0.3, iou_vals_act, width=0.3, zorder=3,
            label='Before class balancing')
    plt.bar(x, iou_vals_bal, width=0.3,
            align='center', zorder=3, label='After class balancing')
    plt.xticks(x, sorted_classes, rotation=80)
    plt.grid(zorder=0, axis='y')

    plt.tick_params(axis='both', which='major', labelsize=set_fonts)
    plt.ylabel('mIOU (%)', fontsize=set_fonts)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=2,
               borderaxespad=0., prop={'size': set_fonts})
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
        print labels_list[index], '=', round(miou[-1], 2)

    plt.tick_params(axis='both', which='major',
                    labelsize=set_fonts)
    plt.xlabel('Number of training steps',
               fontsize=set_fonts, labelpad=12)
    plt.ylabel('mIOU (%)', fontsize=set_fonts)
    plt.legend(plot_objs, labels_list,
               fontsize=set_fonts, loc=0)
    plt.tight_layout()
    plt.grid(zorder=0, linestyle='dotted')
    plt.show()


def plot_confusion(confusion_matrix, label_def):

    fig, ax = plt.subplots()
    confusion_matrix = confusion_matrix/(
        np.sum(confusion_matrix, axis=1)[np.newaxis].T) * 100

    sns.heatmap(confusion_matrix, xticklabels=label_def.values(),
                yticklabels=label_def.values(), vmin=0, vmax=50,
                cmap=sns.color_palette('Blues', n_colors=30),
                annot=True, annot_kws={"size": 12}, fmt='.2f')
    ax.xaxis.tick_top()
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.show()


def plot_quantization_results(miou_list, labels_list, set_fonts,
                              colormap):

    model_size_list = [8.7]*4 + [2.8]*4 + [165.6]*4 + [44.7]*4
    miou = []
    mark = ['o']*4 + ['d']*4 + ['^']*4 + ['s']*4
    colors = [colormap(0.2), colormap(0.5), colormap(0.7), colormap(0.9)]*4
    plot_objs = []

    for mo in miou_list:
        miou.append(mo[-1])
    print(np.round(miou, 2))

    fig, ax = plt.subplots()
    for index, (sz, mo) in enumerate(zip(model_size_list, miou)):
        plot_objs.append(ax.scatter(sz, mo, color=colors[index],
                         marker=mark[index], linewidths=2))
        plt.ylim([60, 102])

    plot_objs = np.reshape(plot_objs, (4, 4))
    plot_objs_t = plot_objs.T

    plot_objs = np.vstack((plot_objs, plot_objs_t))

    new_plots = []

    for obj in plot_objs:
        new_plots.append(tuple(obj))

    rect = patches.Rectangle((0, 90), 10, 10, fill=False,
                             edgecolor='g', linewidth=2)
    ax.add_patch(rect)

    plt.tick_params(axis='both', which='major',
                    labelsize=set_fonts)
    plt.xlabel('Occupied disk memory',
               fontsize=set_fonts, labelpad=12)
    plt.ylabel('mIOU (%)', fontsize=set_fonts)
    leg = plt.legend(new_plots, labels_list, fontsize=set_fonts, loc=4, ncol=2)
    leg.legendHandles[2].set_color(colormap(0.1))
    print(leg.legendHandles)
    plt.tight_layout()
    plt.grid(zorder=0, linestyle='dotted')
    plt.show()


def plot_quant(labels_list, set_fonts, colormap,):

    model_size_list = [8.7] + [2.8] + [165.6] + [44.7]
    infer_time = [0.9811, 1.4560, 5.5325, 7.6256]
    mark = ['o'] + ['d'] + ['^'] + ['s']
    colors = [colormap(0.2), colormap(0.5), colormap(0.7), colormap(0.9)]

    fig, ax = plt.subplots()

    rect = patches.Rectangle((0, 0), 10, 1, fill=False,
                             edgecolor='g', linewidth=2, zorder=0)
    ax.add_patch(rect)

    for index, (siz, ti) in enumerate(zip(model_size_list, infer_time)):
        ax.scatter(siz, ti, color=colors[index], label=labels_list[index],
                   marker=mark[index], linewidths=2)

    plt.tick_params(axis='both', which='major',
                    labelsize=set_fonts)
    plt.ylim([0, 9])
    plt.xlabel('Occupied disk memory',
               fontsize=set_fonts, labelpad=12)
    plt.ylabel('Inference time (s)', fontsize=set_fonts)
    plt.legend(fontsize=set_fonts, loc=4, ncol=2)
    plt.tight_layout()
    plt.grid(zorder=0, linestyle='dotted')
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
    iou_list = []

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
        iou_list.append(cls_to_iou)

        if FLAGS.metric_to_plot == 'class_iou':
            plot_class_ious(cls_to_iou, set_fonts,
                            segmentation_dataset.get_cls_to_percentage(
                                dataset))

        elif FLAGS.metric_to_plot == 'confusion_matrix':
            plot_confusion(confusion_matrix, label_def)

    if FLAGS.metric_to_plot == 'class_balance':
        plot_class_balance(iou_list, set_fonts)

    if FLAGS.metric_to_plot == 'miou':
        plot_mious(miou_list, step_list, FLAGS.dataset, set_fonts,
                   model_variant='mobileNet')

        miou_list = np.array(miou_list)
        step_list = np.array(step_list)
        # labels_list = ['VB: MobileNetv2', 'WB: MobileNetv2', 'VB: Xception', 'WB: Xception']
        # labels_list = ['Real training data', 'VB all training data', 'WB all training data',
        #                'VB artificial training data']
        # labels_list = ['Real training data', 'VB artificial Training data',
        #                'WB all training data']
        #labels_list = ['Real training data', 'Artificial Training data']
        # labels_list = ['PASCAL pretrained', 'Binary pretrained']
        labels_list = ['MobileNetv2: PASCAL VOC 2012', 'MobileNetv2: atWork_binary',
                       'Xception: PASCAL VOC 2012', 'Xception: atWork_binary']
        # labels_list = ['VB: PASCAL pretrained', 'VB: Binary pretrained',
        #                'WB: PASCAL pretrained', 'WB: Binary pretrained']
        # labels_list = ['BW: actual', 'BW: 0.3', 'BW: 0.6', 'BW: 0.9']
        # labels_list = ['cosine restarts', 'poly']

        select_idx = [idx for idx, dataset in enumerate(FLAGS.dataset)
                      if 'full' in dataset]
        # plot_mious(miou_list[select_idx], step_list[select_idx],
        #            labels_list, set_fonts, model_variant='xception')

        # plot_miou_bars(miou_list, set_fonts, labels_list)

    if FLAGS.metric_to_plot == 'quant_results':
        miou_list = np.array(miou_list)
        labels_list = ['MobileNetv2', 'MobileNetv2-8bit', 'Xception', 'Xception-8bit',
                       'atWork_full', 'atWork_size_invariant', 'atWork_similar_shapes',
                       'atWork_binary']
        plot_quantization_results(miou_list,
                                  labels_list, set_fonts, colormap)

    if FLAGS.metric_to_plot == 'plot_quant':
        labels_list = ['MobileNetv2', 'MobileNetv2-8bit', 'Xception', 'Xception-8bit']
        plot_quant(labels_list, set_fonts, colormap)


if __name__ == '__main__':
    flags.mark_flag_as_required('events_dir')
    tf.app.run()
