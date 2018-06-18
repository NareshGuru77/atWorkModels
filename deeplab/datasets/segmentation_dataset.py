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

"""Provides data from semantic segmentation datasets.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""
import collections
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'num_classes',   # Number of semantic classes.
     'ignore_label',  # Ignore label value.
     'labels_to_class',
    ]
)

_ATWORK_BINARY_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 7500,
        'val': 942,
        'test': 939,
        'trainVal': 8442,
        'trainValTest': 9381,
    },
    num_classes=2,
    ignore_label=255,
    labels_to_class={0: 'background', 1: 'foreground'},
)

_ATWORK_SIMILAR_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 7500,
        'val': 942,
        'test': 939,
        'trainVal': 8442,
        'trainValTest': 9381,
    },
    num_classes=13,
    ignore_label=255,
    labels_to_class={0: 'background', 1: 'f_s20_40_20_40_B,G', 2: 'm20_100', 3: 'm20_30',
                     4: 'r20', 5: 'bearing_box', 6: 'bearing', 7: 'axis', 8: 'distance_tube',
                     9: 'motor', 10: 'container', 11: 'em_01', 12: 'em_02'},
)

_ATWORK_SIZE_INVARIANT_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 7500,
        'val': 942,
        'test': 939,
        'trainVal': 8442,
        'trainValTest': 9381,
    },
    num_classes=15,
    ignore_label=255,
    labels_to_class={0: 'background', 1: 'f_s20_40_20_40_B', 2: 'f_s20_40_20_40_G',
                     3: 'm20_100', 4: 'm20_30', 5: 'r20', 6: 'bearing_box', 7: 'bearing',
                     8: 'axis', 9: 'distance_tube', 10: 'motor', 11: 'container_box_blue',
                     12: 'container_box_red', 13: 'em_01', 14: 'em_02'},
)

_ATWORK_FULL_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 7500,
        'val': 942,
        'test': 939,
        'trainVal': 8442,
        'trainValTest': 9381,
    },
    num_classes=19,
    ignore_label=255,
    labels_to_class={0: 'background', 1: 'f20_20_B', 2: 's40_40_B', 3: 'f20_20_G',
                     4: 's40_40_G', 5: 'm20_100', 6: 'm20', 7: 'm30', 8: 'r20',
                     9: 'bearing_box_ax01', 10: 'bearing', 11: 'axis', 12: 'distance_tube',
                     13: 'motor', 14: 'container_box_blue', 15: 'container_box_red',
                     16: 'bearing_box_ax16', 17: 'em_01', 18: 'em_02'}
,
)


_DATASETS_INFORMATION = {
    'atWork_binary': _ATWORK_SIMILAR_INFORMATION,
    'atWork_similar': _ATWORK_SIMILAR_INFORMATION,
    'atWork_size_invariant': _ATWORK_SIMILAR_INFORMATION,
    'atWork_full': _ATWORK_SIMILAR_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_dataset(dataset_name, split_name, dataset_dir):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    split_name: A train/val Split name.
    dataset_dir: The directory of the dataset sources.

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
  ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label
  labels_to_class = _DATASETS_INFORMATION[dataset_name].labels_to_class

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
  keys_to_features = {
          'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'labels_class': tfexample_decoder.Image(
          image_key='image/segmentation/class/encoded',
          format_key='image/segmentation/class/format',
          channels=1),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=ignore_label,
      num_classes=num_classes,
      labels_to_class=labels_to_class,
      name=dataset_name,
      multi_label=True)
