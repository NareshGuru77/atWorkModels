
import os
import numpy as np
import tensorflow as tf
from deeplab.utils import save_annotation
import cv2
import timeit

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('inference_dir', './atWorkData/inference/',
                    'Where to save inference result')

flags.DEFINE_string('image_path', None,
                    'Path of the image to segment')

flags.DEFINE_string('graph_path', None,
                    'Path of the checkpoint')

flags.DEFINE_string('dataset', 'atWork',
                    'Name of the segmentation dataset.')

flags.DEFINE_multi_integer('inference_crop_size', [481, 641],
                           'Crop size [height, width] for visualization.')

flags.DEFINE_bool('print_flops', False, 'Print flops of inference graph')

flags.DEFINE_bool('avg_inf_time', False,
                  'Calculate average inference time over a set of runs')

flags.DEFINE_integer('num_runs', 20,
                     'Number of repetitions for inference if average inference '
                     'time is required')

# The format to save prediction
_PREDICTION_FORMAT = '%s_prediction'

# The format to save prediction
_RAW_FORMAT = '%s'

_INPUT_OP = 'ImageTensor'

_OUTPUT_OP = 'SemanticPredictions'


def load_graph():
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(FLAGS.graph_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def get_prediction_format():
    return _PREDICTION_FORMAT


def get_inference_time():

    pass


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.inference_dir)

    elapsed_time = 0

    g = load_graph()
    with g.as_default(), tf.device("/cpu:0"):

        if FLAGS.print_flops:
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(g, options=opts)
            if flops is not None:
                print 'Total flops: ', flops.total_float_ops

        image_name = FLAGS.image_path.split('/')[-1]
        image_name, image_extension = image_name.split('.')

        supported_extensions = ['png', 'jpeg', 'jpg']

        if not any(image_extension == extension
                   for extension in supported_extensions):
            raise ValueError('Image extension "{}" not supported...'.
                             format(image_extension))

        image = cv2.imread(FLAGS.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_dimensions = image.shape[0:2]
        image = cv2.resize(image, tuple(reversed(FLAGS.inference_crop_size)))
        image = np.expand_dims(image, 0)

        input_operation = g.get_operation_by_name('import/'+_INPUT_OP)
        output_operation = g.get_operation_by_name('import/'+_OUTPUT_OP)

        with tf.Session(graph=g) as sess:

            semantic_predictions = None
            if FLAGS.avg_inf_time:
                for i in range(20):
                    start_time = timeit.default_timer()
                    semantic_predictions = sess.run(output_operation.outputs[0],
                                                    feed_dict={
                                                    input_operation.outputs[0]: image})

                    elapsed_time += timeit.default_timer() - start_time

                elapsed_time = np.round(elapsed_time/20, 4)
            else:
                start_time = timeit.default_timer()
                semantic_predictions = sess.run(output_operation.outputs[0],
                                                feed_dict={
                                                    input_operation.outputs[0]: image})

                elapsed_time = timeit.default_timer() - start_time

        print 'Inference time : {} s'.format(elapsed_time)

        result = np.array(semantic_predictions, dtype=np.uint8)
        result = np.squeeze(result)
        result = cv2.resize(result, tuple(reversed(original_image_dimensions)))

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

    return elapsed_time


if __name__ == '__main__':
    flags.mark_flag_as_required('image_path')
    flags.mark_flag_as_required('graph_path')
    tf.app.run()
