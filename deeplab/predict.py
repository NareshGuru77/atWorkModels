import sys
sys.path.append('..')
sys.path.append('../slim')
sys.path.append('/content/drive/research')
sys.path.append('/content/drive/research/slim')

import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation
from deeplab.utils import get_dataset_colormap

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

flags.DEFINE_multi_integer('eval_crop_size', [513, 513],
                           'Image crop size [height, width] for evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.

flags.DEFINE_string('dataset', 'atWork',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

  with tf.Graph().as_default():
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

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.eval_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    tf.logging.info('Performing single-scale test.')
    predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid)
    predictions = predictions[common.OUTPUT_TYPE]
    #predictions = tf.reshape(predictions, shape=[-1])

    predictions = predictions.eval()
    print(predictions)
    save_annotation.save_annotation(predictions,
                    './results/',
                    samples[common.IMAGE_NAME],
                    add_colormap=True,
                    colormap_type=get_dataset_colormap.get_cityscapes_name())


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
