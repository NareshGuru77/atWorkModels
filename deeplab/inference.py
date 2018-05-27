import sys
sys.path.append('..')
sys.path.append('../slim')
import numpy as np
import tensorflow as tf
from deeplab.datasets import build_data
from deeplab import common
from deeplab import model
from deeplab.utils import save_annotation

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS


flags.DEFINE_string('inference_dir', './atWorkData/inference/',
                    'Where to save inference result')

flags.DEFINE_string('image_path', None,
                    'Path of the image to segment')

flags.DEFINE_string('inference_graph_path', None,
                    'Path of the frozen inference graph')

flags.DEFINE_string('checkpoint_path', None,
                    'Path of the checkpoint')

flags.DEFINE_string('dataset', 'atWork',
                    'Name of the segmentation dataset.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_integer('num_classes', None,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_multi_integer('inference_crop_size', [471, 631],
                           'Crop size [height, width] for visualization.')

# The format to save prediction
_PREDICTION_FORMAT = '%04d_prediction'

# The format to save prediction
_RAW_FORMAT = '%04d_raw'

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.inference_dir)

    g = tf.Graph()
    with g.as_default():

        image_name = FLAGS.image_path.split('/')[-1]
        image_name, image_extension = image_name.split('.')
        image_number = int(image_name)

        supported_extensions = ['png', 'jpeg', 'jpg']

        if not any(image_extension == extension
               for extension in supported_extensions):
            raise ValueError('Image extension "{}" not supported...'.
                             format(image_extension))

        reader = build_data.ImageReader(image_extension)
        image = reader.decode_image(tf.gfile.FastGFile(FLAGS.image_path, 'r').read())
        image = tf.identity(image)
        image.set_shape([None, None, 3])
        image = tf.expand_dims(image, 0)

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
            crop_size= FLAGS.inference_crop_size,
            atrous_rates=None,
            output_stride=FLAGS.output_stride)

        predictions = model.predict_labels(
            image,
            model_options=model_options,
            image_pyramid=None)
        predictions = predictions[common.OUTPUT_TYPE]

        tf.train.get_or_create_global_step()
        saver = tf.train.Saver(slim.get_variables_to_restore())
        sv = tf.train.Supervisor(graph=g,
                                 logdir=FLAGS.inference_dir,
                                 init_op=tf.global_variables_initializer(),
                                 summary_op=None,
                                 summary_writer=None,
                                 global_step=None,
                                 saver=saver)

        with sv.managed_session(start_standard_services=False) as sess:
            sv.start_queue_runners(sess)
            sv.saver.restore(sess, FLAGS.checkpoint_path)
            semantic_predictions = sess.run(predictions)

        result = np.array(semantic_predictions, dtype= np.uint8)
        result = np.squeeze(result)

        # save raw result...
        save_annotation.save_annotation(
            result, FLAGS.inference_dir,
            _RAW_FORMAT % image_number,
            add_colormap=False)

        # save result as color image...
        save_annotation.save_annotation(
            result, FLAGS.inference_dir,
            _PREDICTION_FORMAT % image_number, add_colormap=True,
            colormap_type=FLAGS.dataset)

if __name__ == '__main__':
    flags.mark_flag_as_required('image_path')
    flags.mark_flag_as_required('inference_graph_path')
    flags.mark_flag_as_required('checkpoint_path')
    flags.mark_flag_as_required('num_classes')
    tf.app.run()