from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from kinetics_i3d import i3d


NUM_CLASSES = 600
_CHECKPOINT_PATHS = {
    'rgb': '/kinetics_i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': '/kinetics_i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': '/kinetics_i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '/kinetics_i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '/kinetics_i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 9
tf.compat.v1.disable_eager_execution()
rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

with tf.variable_scope('RGB'):
  rgb_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
  rgb_logits, end_points = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)

rgb_variable_map = {}
for variable in tf.global_variables():
  if variable.name.split('/')[0] == 'RGB':
    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable

rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

model_logits = rgb_logits
model_predictions = tf.nn.softmax(model_logits)


def extract_i3d_feature(new_image, end_points):
  kk = None
  sess = tf.Session()
  rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb600'])
  tf.logging.info('RGB checkpoint restored')
  feed_dict = {}
  new_image =  np.transpose(new_image, (1,0 , 3, 4, 2))
  rgb_sample = new_image
  tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
  feed_dict[rgb_input] = rgb_sample
  out_logits, end_points_ = sess.run(
      [model_logits, end_points],
      feed_dict=feed_dict)

  kk = end_points_['sam'].squeeze()
  sess.close()
  return kk



