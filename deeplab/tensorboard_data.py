
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from deeplab.datasets import segmentation_dataset
from scipy.signal import savgol_filter
import copy
import os

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('dataset', 'atWork_binary',
                          'Name of the segmentation dataset.')

flags.DEFINE_multi_string('events_dir', None,
                          'Directory of the events file.')

flags.DEFINE_integer('final_step', 30000,
                     'Final step of logs in the event file.')


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
                confusion_matrix = np.array(confusion_matrix, dtype=np.uint32)

            if v.tag == 'miou_1.0':
                miou.append(v.simple_value)

    step = np.unique(step)

    return step, miou, cls_to_iou, confusion_matrix, confusion_string


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    for variant, event_dir in zip(FLAGS.dataset, FLAGS.events_dir):

        event_file = os.path.join(event_dir, os.listdir(
            event_dir)[0])
        label_def = segmentation_dataset.get_label_def(variant)
        class_names = np.array(label_def.values())
        class_names = np.array(['class/' + cls for cls in class_names])
        cls_to_iou = {key: [] for key in class_names}

        (step, miou, _,
         _, _) = get_results(event_file, class_names, cls_to_iou)

        if 'size' in variant:
            if len(miou) > 1:
                miou = savgol_filter(miou, 5, 2)
                plt.plot(step, miou, label=variant)
            else:
                plt.scatter(FLAGS.final_step, miou, marker='+',
                            linewidths=2, label='quantized model')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('events_dir')
    tf.app.run()
