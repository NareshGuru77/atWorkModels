
"""
Evalute a frozen inference graph.
"""

import math
import tensorflow as tf
from deeplab import common
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab import eval

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

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

    if FLAGS.eval_batch_size != 1:
        raise ValueError('Batch size {} is not allowed. '
                         'Inference graph can only be '
                         'evaluated image by image.'.format(
                            FLAGS.eval_batch_size))

    batch_size = 1

    g = tf.Graph()
    with g.as_default():

        samples = input_generator.get(
              dataset,
              FLAGS.eval_crop_size,
              batch_size,
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

        (_,
         summary_op,
         metrics_to_updates,
         confusion_matrix,
         category_iou) = eval.create_metrics(g, samples, dataset, predictions)

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

            for image_number in range(dataset.num_samples):
                if ((image_number + 1) % log_steps == 0 or
                        image_number == dataset.num_samples - 1):
                    tf.logging.info('Evaluation [%d/%d]', image_number + 1,
                                    dataset.num_samples)

                sess.run([samples[common.IMAGE], metrics_to_updates.values()])

            sv.summary_computed(sess, sess.run(summary_op))
            sess.run([confusion_matrix, category_iou])


if __name__ == '__main__':
    flags.mark_flag_as_required('graph_path')
    flags.mark_flag_as_required('eval_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
