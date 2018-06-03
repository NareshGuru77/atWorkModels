
import numpy as np
import tensorflow as tf
from deeplab.utils import save_annotation
import cv2

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

# The format to save prediction
_PREDICTION_FORMAT = '%s_prediction'

# The format to save prediction
_RAW_FORMAT = '%s_raw'

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

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.inference_dir)

    g = load_graph()
    with g.as_default():
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

        # input_operation = g.get_operation_by_name('import/'+_INPUT_OP)
        # output_operation = g.get_operation_by_name('import/'+_OUTPUT_OP)

        input_tensor = g.get_tensor_by_name('import/' + _INPUT_OP + ':0')
        output_tensor = g.get_tensor_by_name('import/' + _OUTPUT_OP + ':0')

        with tf.Session(graph=g) as sess:

            # semantic_predictions = sess.run(output_operation.outputs[0],
            #                                 feed_dict={
            #     input_operation.outputs[0]: image
            # })

            semantic_predictions = sess.run(output_tensor,
                                            feed_dict={
                                                input_tensor: image
                                            })

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

if __name__ == '__main__':
    flags.mark_flag_as_required('image_path')
    flags.mark_flag_as_required('graph_path')
    tf.app.run()