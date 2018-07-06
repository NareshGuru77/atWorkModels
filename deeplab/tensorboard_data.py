
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from deeplab.datasets import segmentation_dataset
from scipy.signal import savgol_filter
import copy
import os
import seaborn as sns

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('dataset', 'atWork_binary',
                          'Name of the segmentation dataset.')

flags.DEFINE_multi_string('events_dir', None,
                          'Directory of the events file.')

flags.DEFINE_integer('final_step', 30000,
                     'Final step of logs in the event file.')

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
                cls_to_iou[v.tag].append(v.simple_value)

            if event.step == FLAGS.final_step and v.tag == 'confusion_matrix':
                confusion_string = np.array(v.tensor.string_val)
                confusion_string = np.reshape(confusion_string,
                                              (confusion_rc_len,
                                               confusion_rc_len))
                confusion_matrix = confusion_string[:, 1:confusion_rc_len][
                                   1:confusion_rc_len, :]
                confusion_matrix = np.array(confusion_matrix, dtype=np.float32)

            if v.tag == 'miou_1.0':
                miou.append(v.simple_value)

    step = np.unique(step)

    return step, miou, cls_to_iou, confusion_matrix, confusion_string


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels_mn = ['mobileNet']*3
    labels_xc = ['transfer']*3
    labels = labels_mn+labels_xc

    cls_to_iou = {}
    step = []

    for index, (variant, event_dir) in enumerate(zip(FLAGS.dataset, FLAGS.events_dir)):

        event_file = os.path.join(event_dir, os.listdir(
            event_dir)[0])
        label_def = segmentation_dataset.get_label_def(variant)
        class_names = np.array(label_def.values())
        class_names = np.array(['class/' + cls for cls in class_names])
        cls_to_iou = {key: [] for key in class_names}

        (step, miou, cls_to_iou,
         confusion_matrix, _) = get_results(event_file, class_names, cls_to_iou)

        correct_predictions = np.diag(confusion_matrix)
        second_max = sorted(correct_predictions)[-2]
        confusion_matrix = confusion_matrix/np.max(confusion_matrix, axis=0)
        confusion_matrix = np.round(confusion_matrix, 2)

        sns.heatmap(confusion_matrix, xticklabels=label_def.values(),
                    yticklabels=label_def.values(), vmin=0, vmax=0.6,
                    cmap=sns.color_palette('Blues', n_colors=30),
                    annot=True, annot_kws={"size": 12})
        plt.xticks(rotation=80)
        plt.tight_layout()
        plt.show()

    #     if 'size' in variant:
    #         if len(miou) > 1:
    #             miou = savgol_filter(miou, 5, 2)
    #             plt.plot(step, miou, label=labels[index] + ' : ' + variant)
    #             plt.text(step[-1], miou[-1], round(miou[-1], 4))
    #         else:
    #             plt.scatter(FLAGS.final_step, miou, marker='+',
    #                         linewidths=2, label='quantized model: ' + labels[index])
    #             plt.text(FLAGS.final_step, miou[0], round(miou[0], 4))
    #
    # plt.legend()
    # plt.show()

    # for index, (key, values) in enumerate(cls_to_iou.items()):
    #     values = savgol_filter(values, 7, 3)
    #     plt.plot(step, values, label=key, c=np.flip(colormap[index], 0)/255.)
    #
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('events_dir')
    tf.app.run()
