
import numpy as np
import tensorflow as tf
from deeplab.datasets import build_data
from deeplab import common
from deeplab import model
from deeplab.utils import save_annotation
import cv2

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('inference_dir', './atWorkData/inference/',
                    'Where to save inference result')

flags.DEFINE_string('image_path', None,
                    'Path of the image to segment')

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

flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

# The format to save prediction
_PREDICTION_FORMAT = '%s_prediction'

# The format to save prediction
_RAW_FORMAT = '%s_raw'


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.inference_dir)

    g = tf.Graph()
    with g.as_default():
        image_name = FLAGS.image_path.split('/')[-1]
        image_name, image_extension = image_name.split('.')

        supported_extensions = ['png', 'jpeg', 'jpg']

        if not any(image_extension == extension
                   for extension in supported_extensions):
            raise ValueError('Image extension "{}" not supported...'.
                             format(image_extension))

        reader = build_data.ImageReader(image_extension)
        image = reader.decode_image(tf.gfile.FastGFile(FLAGS.image_path, 'r').read())
        image = tf.identity(image)
        original_image_dimensions = image.get_shape().as_list()[0:2]
        original_image_dimensions = reversed(original_image_dimensions)

        image = tf.image.resize_images(
            image, [480, 640], method=tf.image.ResizeMethod.BILINEAR,
            align_corners=True)
        image.set_shape([None, None, 3])
        image = tf.expand_dims(image, 0)

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
            crop_size=FLAGS.inference_crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        predictions = model.predict_labels(
            image,
            model_options=model_options,
            image_pyramid=None)
        predictions = predictions[common.OUTPUT_TYPE]
        # predictions = tf.image.resize_images(
        #     predictions, original_image_dimensions,
        #     method=tf.image.ResizeMethod.BILINEAR,
        #     align_corners=True)

        param_stats = tf.profiler.profile(
            tf.get_default_graph(),
            options=tf.profiler.ProfileOptionBuilder
                .trainable_variables_parameter())
        print('Total parameters: ',param_stats.total_parameters)

        total_parameters = 0
        for variable in tf.trainable_variables():

            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total parameters: ', total_parameters)

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


        result = np.array(semantic_predictions, dtype=np.uint8)
        result = np.squeeze(result)
        result = cv2.resize(result, tuple(original_image_dimensions))

        # save raw result...
        save_annotation.save_annotation(
            result, FLAGS.inference_dir,
            _RAW_FORMAT % image_name,
            add_colormap=False)

        # save result as color image...
        save_annotation.save_annotation(
            result, FLAGS.inference_dir,
            _PREDICTION_FORMAT % image_name, add_colormap=True,
            colormap_type=FLAGS.dataset)


if __name__ == '__main__':
    flags.mark_flag_as_required('image_path')
    flags.mark_flag_as_required('checkpoint_path')
    flags.mark_flag_as_required('num_classes')
    tf.app.run()
