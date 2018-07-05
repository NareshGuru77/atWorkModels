
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from deeplab.datasets import segmentation_dataset

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'atWork_binary',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('events_path', None,
                    'Path to events file.')

flags.DEFINE_integer('final_step', 30000,
                     'Final step of logs in the event file.')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    step = []

    label_def = segmentation_dataset.get_label_def(FLAGS.dataset)
    class_names = np.array(label_def.values())
    class_names = np.array(['class/' + cls for cls in class_names])
    loss_dict = {key: [] for key in class_names}
    confusion_rc_len = len(class_names) + 1

    confusion_string = None
    confusion_matrix = None
    for event in tf.train.summary_iterator(FLAGS.events_path):
        step.append(event.step)
        for v in event.summary.value:
            if np.any(class_names == v.tag):
                loss_dict[v.tag].append(v.simple_value)

            if event.step == FLAGS.final_step and v.tag == 'confusion_matrix':
                confusion_string = np.array(v.tensor.string_val)
                confusion_string = np.reshape(confusion_string,
                                              (confusion_rc_len,
                                               confusion_rc_len))
                confusion_matrix = confusion_string[:, 1:confusion_rc_len][
                                   1:confusion_rc_len, :]
                confusion_matrix = np.array(confusion_matrix, dtype=np.uint32)

    print('confusion matrix with class names: ', confusion_string)
    print('confusion matrix: ', confusion_matrix)

    step = np.unique(step)
    for key, value in loss_dict.items():
        plt.plot(step, value, label=key)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('events_path')
    tf.app.run()
