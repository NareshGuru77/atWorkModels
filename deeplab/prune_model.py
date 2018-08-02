

import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pruning_hparams', '',
    """Comma separated list of pruning-related hyperparameters""")

flags.DEFINE_string('pruning_logs_dir', './checkpoints_logs/pruning/',
                    'Where to save inference result')

flags.DEFINE_string('frozen_graph_path', './atWorkData/inference/mobileNet_binary/mobilenet.pb',
                    'Path of the frozen inference graph')

flags.DEFINE_string('checkpoint_path', './checkpoints_logs/train_logs/mobileNet/intermediate_logs/similar_01/model.ckpt-30000',
                    'Path of the checkpoint')

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    g = tf.Graph()
    with g.as_default():

        # Create global step variable
        global_step = tf.train.get_or_create_global_step()

        # Parse pruning hyperparameters
        pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)
        print (pruning_hparams)
        # Create a pruning object using the pruning specification
        p = pruning.Pruning(spec=pruning_hparams, global_step=global_step)

        # Add conditional mask update op. Executing this op will update all
        # the masks in the graph if the current global step is in the range
        # [begin_pruning_step, end_pruning_step] as specified by the pruning spec
        mask_update_op = p.conditional_mask_update_op()

        # Add summaries to keep track of the sparsity in different layers during training
        p.add_pruning_summaries()

        saver = tf.train.Saver(slim.get_variables_to_restore(exclude=['model_pruning']))
        sv = tf.train.Supervisor(graph=g,
                                 logdir=FLAGS.pruning_logs_dir,
                                 init_op=tf.global_variables_initializer(),
                                 summary_op=None,
                                 summary_writer=None,
                                 global_step=None,
                                 saver=saver)

        with sv.managed_session(start_standard_services=False) as sess:
            sv.start_queue_runners(sess)
            sv.saver.restore(sess, FLAGS.checkpoint_path)
            # Update the masks by running the mask_update_op
            sess.run(mask_update_op)



if __name__ == '__main__':
    tf.app.run()