"""
Helper functions to read and analyse files in the interfile format
"""
import math
import os
import re
import pylab
import numpy as np
from scipy.ndimage import gaussian_filter


def parse_raw_file(file_name):
  #parse qetir raw files without header
  return {
      "type": np.float32,
      'flip': False,
      'scaling_factor_xy': 2.5,
      'scaling_factor_z': 2.5,
      'size': (161, 161, 161),
      'path_to_data_file': file_name
  }


def is_correct_number_format(param):
  try:
    return (
        param['!number format'] == 'float'
        or param['!number format'] == 'short float'
    ) and param['!number of bytes per pixel'] == '4'
  except KeyError:
    return False


def interfile_parser(file_name):
  """
    Parse interfile header to dict
    :param file_name: str interfile file name
    :return:
    """
  is_raw_file = lambda file_name: file_name.split(".")[-1] == 'raw'

  if is_raw_file(file_name):
    return parse_raw_file(file_name)

  param = {}
  with open(file_name, 'r', encoding="utf-8") as f:
    try:
      for line in f.readlines():
        match_obj = re.match(r'(.*) := (.*)', line, re.M | re.I)
        if match_obj:
          param[match_obj.group(1)] = match_obj.group(2)
    except:
      raise ValueError("Unsupported format") from Exception
    try:
      param['size'] = (
          int(param['!matrix size [3]']), int(param['!matrix size [2]']),
          int(param['!matrix size [1]'])
      )
    except KeyError:
      raise Exception("Bad parsing Matrix size") from KeyError
    try:
      param['flip'] = bool(param['flip'])
    except KeyError:
      param['flip'] = False  # add fil for QETRI

    if is_correct_number_format(param):
      param["type"] = np.float32
    else:
      raise ValueError("Bad number format") from Exception

    try:
      param['scaling_factor_xy'] = float(
          param['scaling factor (mm/pixel) [1]']
      )
      param['scaling_factor_z'] = float(param['scaling factor (mm/pixel) [3]'])
    except KeyError:
      raise KeyError("Bad parsing scaling_factor") from Exception
    if param.get('name of data file'):
      param['path_to_data_file'] = os.path.join(
          os.path.dirname(file_name), param['name of data file']
      )
    elif param.get('!name of data file'):
      param['path_to_data_file'] = os.path.join(
          os.path.dirname(file_name), param['!name of data file']
      )

  return param


def get_image(interfile_header, params=None):
  image_parameters = interfile_parser(interfile_header)
  image = interfile2array(image_parameters)
  # smooth image if needed
  if params:
    if "sigma_smoothing" in params:
      sigma = params["sigma_smoothing"]
      image = gaussian_filter(image, sigma)
  return image


def read_parameters(parameters):
  scaling_factor_xy = parameters['scaling_factor_xy']
  scaling_factor_z = parameters['scaling_factor_z']
  size_xy = parameters['size'][1]
  size_z = parameters['size'][0]
  return scaling_factor_xy, scaling_factor_z, size_xy, size_z


def interfile2array(params):
  """
    convert interfile to numpy array
    :param param: dict with keys: 'path_to_data_file',
                  'type' corresponding to data type,
                  and 'size' corresponding to matrix size.
    :return: numpy.array
  """
  resh_arr = np.fromfile(params['path_to_data_file'], dtype=params['type'])
  resh_arr = resh_arr.reshape(params['size'])
  resh_arr = resh_arr[:, ::-1, ::-1]
  return resh_arr


def get_circle_measure(
    array, x0, y0, z0, radius, scaling_factor_xy, scaling_factor_z, size_xy,
    size_z
):
  """
    Get mean and std from sphere (x0, y0, z0) [mm]  radius  [mm], scaling_factor mm/pixel, from array with specfic size
    """

  def cm2pix(dm, dx=scaling_factor_xy, size=size_xy):
    return int(size / 2 + (dm * (1 / dx)))

  x0 = cm2pix(x0)
  y0 = cm2pix(-y0)
  z0 = cm2pix(z0, scaling_factor_z, size_z)
  out_list = []
  radius_xy = int(radius * (1 / scaling_factor_xy))

  y, x = np.ogrid[-radius_xy:radius_xy + 1, -radius_xy:radius_xy + 1]
  index = x**2 + y**2 <= radius_xy**2
  test_array = array
  out_list.extend(
      array[z0, y0 - radius_xy:y0 + radius_xy + 1,
            x0 - radius_xy:x0 + radius_xy + 1][index].flatten()
  )
  test_array[z0, y0 - radius_xy:y0 + radius_xy + 1,
             x0 - radius_xy:x0 + radius_xy + 1][index] = 0
  return np.mean(out_list), np.std(out_list), len(out_list), test_array


def get_sphere_maximum(
    array, x0, y0, z0, radius, scaling_factor_xy, scaling_factor_z, size_xy,
    size_z
):
  """
    Get max  from sphere (x0, y0, z0) [mm]  radius  [mm], scaling_factor mm/pixel, from array with specfic size
    """

  def cm2pix(dm, dx=scaling_factor_xy, size=size_xy):
    return int(size / 2 + (dm * (1 / dx)))

  x0 = cm2pix(x0)
  y0 = cm2pix(-y0)
  z0 = cm2pix(z0, scaling_factor_z, size_z)
  out_list = []
  z = z0
  if np.abs(np.abs(z0 - z) * scaling_factor_z / radius) < 10e-100:
    radius_xy = radius
  else:
    radius_xy = np.abs(z0 - z) * scaling_factor_z * 1 / math.tan(
        math.asin(np.abs(z0 - z) * scaling_factor_z / radius)
    )
  if math.isnan(radius_xy):
    radius_xy = radius

  radius_xy = int(radius_xy * (1 / scaling_factor_xy))

  y, x = np.ogrid[-radius_xy:radius_xy + 1, -radius_xy:radius_xy + 1]
  index = x**2 + y**2 <= radius_xy**2
  out_list.extend(
      array[z, y0 - radius_xy:y0 + radius_xy + 1,
            x0 - radius_xy:x0 + radius_xy + 1][index].flatten()
  )
  return max(out_list)


def show_xy(_array, r=437.3, title=""):
  """
    Show and save slice with normalize (max value as 1.0)
    """
  pylab.figure()
  ############# code for show image with STIR-TOF ring
  # def cropND(img, bounding):
  #     start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
  #     end = tuple(map(operator.add, start, bounding))
  #     slices = tuple(map(slice, start, end))
  #     return img[slices]

  # # foo[foo == 0] = m
  # size = np.shape(_array)
  # phantom_maximum = max(cropND(_array, (size[0]/2,size[1]/2)).flatten())
  # _array[_array > phantom_maximum]= phantom_maximum

  phantom_maximum = max(_array.flatten())
  _array = (_array - min(_array.flatten()))
  _array = (_array / phantom_maximum)
  pylab.xlabel('X [mm]')
  pylab.ylabel('Y [mm]')
  pylab.xlim([-r, r])
  pylab.imshow(_array, cmap="hot", origin='upper', extent=[-r, r, -r, r])
  fig_name = os.path.basename(title).split('.')[0]
  pylab.colorbar()
  pylab.savefig(fig_name)
  # pylab.show()


def get_interfiles(interfile_path):
  file_paths = []
  for interfile_header in os.listdir(interfile_path):
    if interfile_header.endswith(".hdr"):
      file_paths.append(interfile_path + interfile_header)

  #sort file_paths based on iteration number
  file_paths.sort(
      key=lambda x: int(x.split("it")[-1].lstrip().replace(".hdr", ""))
  )

  return file_paths
