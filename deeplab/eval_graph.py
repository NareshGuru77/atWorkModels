# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

import six
import math
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
import numpy as np

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [481, 641],
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')

flags.DEFINE_string('graph_path', None,
                    'Path of the checkpoint')


_INPUT_TENSOR = 'ImageTensor:0'

_OUTPUT_TENSOR = 'SemanticPredictions:0'


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  g = tf.Graph()
  with g.as_default():

    samples = input_generator.get(
          dataset,
          FLAGS.eval_crop_size,
          FLAGS.eval_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          dataset_split=FLAGS.eval_split,
          is_training=False,
          model_variant=FLAGS.model_variant)

    graph_def = tf.GraphDef()
    with open(FLAGS.graph_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    samples[common.IMAGE] = tf.cast(samples[common.IMAGE], tf.uint8)

    tf.import_graph_def(graph_def, input_map={_INPUT_TENSOR:
                                                  samples[common.IMAGE]})

    predictions = g.get_tensor_by_name('import/' + _OUTPUT_TENSOR)
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'
    for eval_scale in FLAGS.eval_scales:
      predictions_tag += '_' + str(eval_scale)
    if FLAGS.add_flipped_images:
      predictions_tag += '_flipped'

    # Define the evaluation metric.
    metric_map = {}
    metric_map[predictions_tag] = tf.metrics.mean_iou(
        predictions, labels, dataset.num_classes, weights=weights)

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))

    for metric_name, metric_value in six.iteritems(metrics_to_values):
      slim.summaries.add_scalar_summary(
          metric_value, metric_name, print_summary=True)

    num_batches = int(
        math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

    tf.logging.info('Eval num images %d', dataset.num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.eval_batch_size, num_batches)

    confusion_tensor = g.get_tensor_by_name("mean_iou/total_confusion_matrix:0")
    confusion_matrix = tf.Print(confusion_tensor, [confusion_tensor],
                                summarize=dataset.num_classes * dataset.num_classes,
                                message='Confusion Matrix')

    category_iou_tensor = g.get_tensor_by_name("mean_iou/div:0")
    category_iou = tf.Print(category_iou_tensor, [category_iou_tensor],
                            summarize=dataset.num_classes,
                            message='Category IOU')

    for index, t in enumerate(tf.unstack(category_iou_tensor)):
        slim.summaries.add_scalar_summary(
            t, 'class/' + dataset.labels_to_class[index], print_summary=False)

    class_names = tf.identity(list(dataset.labels_to_class.values()))
    class_names = tf.reshape(class_names, (dataset.num_classes, 1))
    confusion_string = tf.concat([class_names, tf.as_string(confusion_tensor,
                                                            precision=0)], 1)
    class_names = tf.reshape(class_names, (1, dataset.num_classes))
    empty_column_name = tf.identity('...confusion...')
    empty_column_name = tf.reshape(empty_column_name, (1, 1))
    append_empty_column = tf.concat([empty_column_name, class_names], 1)
    confusion_string = tf.concat([append_empty_column, confusion_string], 0)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.text('confusion_matrix', confusion_string))
    summary_op = tf.summary.merge(list(summaries))

    num_batches = int(math.ceil(
        dataset.num_samples / float(FLAGS.eval_batch_size)))

    tf.train.get_or_create_global_step()

    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.eval_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             global_step=None,
                             saver=None)

    log_steps = int(math.floor(dataset.num_samples/10))

    with sv.managed_session(start_standard_services=False) as sess:
        sv.start_queue_runners(sess)

        for batch in range(num_batches):
            if (batch + 1) % log_steps == 0 or batch == num_batches - 1:
                tf.logging.info('Evaluation [%d/%d]', batch + 1, num_batches)

            sess.run([samples[common.IMAGE], metrics_to_updates.values()])

        sv.summary_computed(sess, sess.run(summary_op))
        sess.run([confusion_matrix, category_iou])


if __name__ == '__main__':
  flags.mark_flag_as_required('graph_path')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
