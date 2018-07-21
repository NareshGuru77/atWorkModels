
import numpy as np

# Dataset names.
_ATWORK = 'atWork'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _ATWORK: 19,
}


def create_atWork_label_colormap():

    colormap = np.asarray([[128, 128, 128], [75, 25, 230], [75, 180, 60],
                           [25, 225, 255], [200, 130, 0], [48, 130, 245],
                           [180, 30, 145], [240, 240, 70], [230, 50, 240],
                           [60, 245, 210], [128, 128, 0], [255, 190, 230],
                           [40, 110, 170], [0, 0, 128], [195, 255, 170],
                           [0, 128, 128], [180, 215, 255], [128, 0, 0],
                           [0, 0, 0], [255, 255, 255]])
    return colormap


def get_atWork_name():
    return _ATWORK


def bit_get(val, idx):
  """Gets the bit value.

  Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.

  Returns:
    The "idx"-th bit of input val.
  """
  return (val >> idx) & 1


def create_label_colormap(dataset=_ATWORK):
  """Creates a label colormap for the specified dataset.

  Args:
    dataset: The colormap used in the dataset.

  Returns:
    A numpy array of the dataset colormap.

  Raises:
    ValueError: If the dataset is not supported.
  """
  if dataset == _ATWORK:
    return create_atWork_label_colormap()
  else:
    raise ValueError('Unsupported dataset.')


def label_to_color_image(label, dataset=_ATWORK):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.
    dataset: The colormap used in the dataset.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= _DATASET_MAX_ENTRIES[dataset]:
    raise ValueError('label value too large.')

  colormap = create_label_colormap(dataset)
  return colormap[label]
