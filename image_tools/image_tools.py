#!/usr/bin/env python3
""" Several methods to load an image and generate some X,Y,Z and XY, XZ, YZ profiles and estimate Full Width Half Maximum of the peak.
  The calculations of fwhm are done according to NEMA standard, using linear approximation and using Gaussian fit, respectively.
"""

import argparse
import logging
import os
import sys
import lzma

from math import floor
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
import gatetools as gt

LOGGER = logging.getLogger(__name__)


def get_info_from_interfile_header(header_file):
  """Gets information from the interfile header file.

  Args:
    header_file (str): Name of the input  header file in the interfile format,
                       alternatively compressed with lzma.
  Returns:
    tuple: f_name (str), voxel_size(list), matrix_size (list), dtype(str)
    with the following meaning:
    f_name is the name of the image file in the interfile format,
    voxel_size is the list of voxel sizes in mm along X,Y and Z axis,
    matrix_size is the list corresponding to the image size along X,Y and Z direction.
    dtype is the string corresponding to the numpy format of numbers in the image matrix

  Examples:
    returned values:  "recon_true_230ps_it1.img", [2.4, 2.4, 2.4], [220, 220, 280], 'float32'

  """
  f_name = ''
  matrix_size = [0, 0, 0]
  voxel_size = [0., 0., 0.]

  if os.path.splitext(header_file)[1] == '.xz':
    my_open = lzma.open
  else:
    my_open = open
  with my_open(header_file, 'rt', encoding="utf-8") as my_file:
    lines = my_file.readlines()
    for line in lines:
      splitted = line.split(' := ')
      splitted[0] = " ".join(splitted[0].split())
      if splitted[0] == '!name of data file':
        f_name = splitted[1].rstrip("\n")
      elif splitted[0].startswith(
          '!matrix size [') and splitted[0][-1] == ']' and splitted[0][-2] in (
              '1', '2', '3'):
        matrix_size[int(splitted[0][-2]) - 1] = int(splitted[1])
      elif splitted[0].startswith('scaling factor (mm/pixel) [') and splitted[
          0][-1] == ']' and splitted[0][-2] in ('1', '2', '3'):
        voxel_size[int(splitted[0][-2]) - 1] = float(splitted[1])
      elif splitted[0] == '!number format':
        if 'float' in splitted[1]:
          ttype = 'float'
        else:
          ttype = 'int'
      elif splitted[0] == '!number of bytes per pixel':
        nbytes = 8 * int(splitted[1])

    dtype = ttype + str(nbytes)

  LOGGER.debug("\tData file name: %s", f_name)
  LOGGER.debug("\tImage dimensions: %s", str(matrix_size))
  LOGGER.debug("\tVoxel size: %s", str(voxel_size))

  return f_name, voxel_size, matrix_size, dtype


def load_image_and_metadata(
    header_file, las_convention=False, return_dicom_properties: bool = False
):
  """Gets metadata and the image file based on the information in the header file.
  Args:
    header_file (str): Name of the input  header file in the interfile format,
                       alternatively compressed with lzma.
    las_convention (bool): If set to true try to load the image in the radiological LAS convention (Left Anterior Superior).
                           Uses C matrix ordering convention and flips the image. Default is set to false.
    return_dicom_properties If set to True return tuple (image_matrix, dicom_properties)
                            where dicom propertis is a tuple (matrix_size, voxel_size, dicom_origin)
                            if set to False (default option) return tuple (image_matrix, voxel_size, matrix_size)

  Returns:
    tuple: image_matrix (numpy.array), voxel_size(list), matrix_size (list)   if return_dicom_properties = False
    tuple  image_matrix (numpy.array), dicom_properties (touple)              if return_dicom_properties = True


    with the following meaning:
    dicom_properties is tuple: matrix_size, voxel_size, dicom_origin
    image_matrix is the numpy.array 3-D matrix contaning the image intensities,
    voxel_size is the list of voxel sizes in mm along X,Y and Z axis,
    matrix_size is the list corresponding to the image size along X,Y and Z direction.
    dicom origin is np.array which corresponds to origin due to dicom standard


  Examples:
    returned values: numpy.array, [2.4, 2.4, 2.4], [220, 220, 280]    if return_dicom_properties = False
    returned values: numpy.array, ([220, 220, 280], [2.4, 2.4, 2.4], [-108.8,-108.8,-138.8]) if return_dicom_properties = True
  """
  load_order = 'F'
  if las_convention:
    load_order = 'C'

  path = os.path.dirname(os.path.abspath(header_file))
  image_file, voxel_size, matrix_size, data_type = get_info_from_interfile_header(
      header_file
  )
  image_file = os.path.join(path, image_file)
  if las_convention:
    matrix_size = np.flip(matrix_size)
  if os.path.splitext(header_file)[1] == '.xz':
    image_file = image_file + ".xz"
    with lzma.open(image_file, "rb") as image_data:
      # https://github.com/numpy/numpy/issues/10866
      image_matrix = np.frombuffer(image_data.read(), dtype=data_type)
  else:
    image_matrix = np.fromfile(image_file, dtype=data_type)
  image_matrix = image_matrix.reshape(matrix_size, order=load_order)
  if las_convention:
    image_matrix = np.flip(image_matrix)

  if return_dicom_properties is False:
    return (image_matrix, voxel_size, matrix_size)

  dicom_origin = get_dicom_origin(matrix_size, voxel_size, img_center=None)
  dicom_properties_object = gt.dicom_properties()
  dicom_properties_object.spacing = voxel_size  # only if gap = 0; probably don't work propely in MRI \
  dicom_properties_object.origin = dicom_origin
  dicom_properties_object.image_shape = matrix_size

  return (image_matrix, dicom_properties_object)


def get_dicom_origin(vector_size, vector_voxel_size, img_center=None):
  """get dicom orgin based for image size, voxel_size, and image center.
  Args:
        vector_size (np.array): 3-elements vector;size of image in voxels
        vector_voxel_size (np.array): 3-elements vector; size of voxels in mm
        center(np.array | list | tuple): 3-elements vector; center of the lab coordinate system in voxels,
                                         default value is the image centre
  Returns:
        numpy.array (lu_voxel-center) : 3-elements vector; position in mm of the first voxel center [0,0,0]
                                        in the lab coordinate system.

  Examples:
    returned values: [-30.0,-15.5,20.2]
  """
  img_size = np.array(vector_size)
  voxel_size = np.array(vector_voxel_size)
  center_mm = 0

  lu_voxel_center_mm = np.divide(voxel_size, 2.0)

  if img_center is None:

    center_mm = (img_size / 2.0) * voxel_size
  else:
    center_mm = (img_center) * voxel_size
  return lu_voxel_center_mm - center_mm


def transform_interfile_header(input_header, output_header, transform_list):
  """Copy an interfile header to another file and edit some of its field on the way.

  Args:
    input_header (str): Name of the input header file in the interfile format.
    output_header (str): Name of the output header file.
    transform_list (list(tuple(str, str))): list of (field, value) tuples.

  Example:
    transform_interfile_header('xcat.hdr', 'xcat_rescaled.hdr', ['data rescale slope', str(2)])
  """
  with open(input_header, 'r', encoding='utf-8') as input_file, open(
      output_header, 'w', encoding='utf-8') as output_file:
    input_header_lines = input_file.readlines()
    for line in input_header_lines:
      splitted = line.split(' := ')
      splitted[0] = " ".join(splitted[0].split())
      for t in transform_list:
        if t[0] in splitted[0]:
          output_file.write(f'{t[0]} := {t[1]}' + os.linesep)
          break
      else:
        output_file.write(line)


def find_voxels_with_maximum_values(image):
  return np.where(np.amax(image) == image)


def get_profile_xy(image, first=None, last=None):
  if not first:
    first = 0
  if not last:
    last = image.shape[2]
  return np.sum(image[:, :, first:last], axis=2)


def get_profile_xz(image, first=None, last=None):
  if not first:
    first = 0
  if not last:
    last = image.shape[1]
  return np.sum(image[:, first:last, :], axis=1)


def get_profile_yz(image, first=None, last=None):
  if not first:
    first = 0
  if not last:
    last = image.shape[0]
  return np.sum(image[first:last, :, :], axis=0)


def get_profile_x_3d(
    image, first_y=None, last_y=None, first_z=None, last_z=None
):
  if not first_y:
    first_y = 0
  if not last_y:
    last_y = image.shape[1]

  if not first_z:
    first_z = 0
  if not last_z:
    last_z = image.shape[2]

  return np.sum(image[:, first_y:last_y, first_z:last_z], axis=(1, 2))


def get_profile_y_3d(
    image, first_x=None, last_x=None, first_z=None, last_z=None
):
  if not first_x:
    first_x = 0
  if not last_x:
    last_x = image.shape[0]

  if not first_z:
    first_z = 0
  if not last_z:
    last_z = image.shape[2]

  return np.sum(image[first_x:last_x, :, first_z:last_z], axis=(0, 2))


def get_profile_z_3d(
    image, first_x=None, last_x=None, first_y=None, last_y=None
):
  if not first_x:
    first_x = 0
  if not last_x:
    last_x = image.shape[0]

  if not first_y:
    first_y = 0
  if not last_y:
    last_y = image.shape[1]
  return np.sum(image[first_x:last_x, first_y:last_y, :], axis=(0, 1))


def get_profile_x(image_2d, first=None, last=None):
  if not first:
    first = 0
  if not last:
    last = image_2d.shape[1]
  return np.sum(image_2d[:, first:last], axis=1)


def get_profile_y(image_2d, first=None, last=None):
  if not first:
    first = 0
  if not last:
    last = image_2d.shape[0]
  return np.sum(image_2d[first:last, :], axis=0)


def get_bin_by_position(pos, voxel_size, profile_len, units="cm"):
  """
  This function returns ``bin`` of ``pos`` that is declared in ``units``.
  It assumes, that whole profile length is ``profile_len`` in bins, 1 bin has ``voxel_size``,
  and position is given assuming 0 is the center of profile.
  """
  unit_scaling = 1
  if units == "mm":
    unit_scaling = voxel_size
  elif units == "cm":
    unit_scaling = voxel_size * 0.1
  # position with repect to 0 is profile_len/2 * units + pos
  # bin number is position/units and rounded
  curr_bin = floor(float(profile_len) / 2 + float(pos) / unit_scaling)
  if curr_bin < 0:
    LOGGER.warning(
        "\tbin is smaller than zero:%i, we set it to zero", curr_bin
    )
    curr_bin = 0
  if curr_bin >= profile_len:
    LOGGER.warning(
        "\tbin is larger than max bin:%i, we set it to max bin", curr_bin
    )
    curr_bin = profile_len - 1
  return curr_bin


def get_position_by_bin(bin_pos, voxel_size, profile_len, units="mm"):
  """
  This function returns ``pos`` of ``bin`` that is declared in ``units``.
  It assumes, that whole profile length is ``profile_len`` in bins, 1 bin has ``voxel_size``,
  and position is given assuming 0 is the center of profile.
  """
  unit_scaling = 1
  if units == "mm":
    unit_scaling = voxel_size
  elif units == "cm":
    unit_scaling = voxel_size * 0.1

  profile_length_scaled = unit_scaling * float(profile_len)
  axis_center_adjustment = -0.5 * profile_length_scaled

  # We add 0.5 * unit_scaling, because we want to get the position corresponding to the center of the bin
  return unit_scaling * bin_pos + 0.5 * unit_scaling + axis_center_adjustment


# assuming voxels_size in mm
def plot_1d_profile(
    profile,
    voxel_size,
    units="voxels",
    title="",
    out_file="",
    disable_display=False
):
  bin_values = [
      get_position_by_bin(x, voxel_size, len(profile), units)
      for x in range(len(profile))
  ]
  plt.figure()
  plt.plot(bin_values, profile)
  plt.xlabel(title + " [" + units + "]")
  if out_file:
    plt.savefig(out_file)
  if not disable_display:
    plt.show()

  print_fwhm(bin_values, profile)


# assuming voxels_size in mm
def plot_2d_profile(
    profile,
    voxel_size,
    units="voxels",
    title="",
    out_file="",
    disable_display=False
):
  if title == "":
    title = ["", ""]
  bins_h_values = [
      get_position_by_bin(x, voxel_size[0], profile.shape[0], units)
      for x in range(profile.shape[0])
  ]

  bins_v_values = [
      get_position_by_bin(x, voxel_size[1], profile.shape[1], units)
      for x in range(profile.shape[1])
  ]
  plt.matshow(
      profile,
      extent=(
          bins_h_values[0], bins_h_values[-1], bins_v_values[0],
          bins_v_values[-1]
      ),
      cmap=plt.cm.gist_rainbow
  )
  plt.colorbar()
  plt.xlabel(title[1] + " [" + units + "]")
  plt.ylabel(title[0] + " [" + units + "]")
  if out_file:
    plt.savefig(out_file)
  if not disable_display:
    plt.show()


def get_out_files(out_file_base):
  out_files = ["" for _ in range(6)]
  if out_file_base:
    out_files = [
        "_profile_x", "_profile_y", "_profile_z", "_profile_xy", "_profile_xz",
        "_profile_yz"
    ]
    out_files = [out_file_base + name + '.png' for name in out_files]
  return out_files


def find_half(x_set, y_set, amp=None, to_left=True):
  assert len(x_set) <= len(y_set)
  _, mean_index, amplitude = find_mean(x_set, y_set)
  if amp is None:
    amp = amplitude
  left_x = mean_index
  left_y = amp
  right_x = mean_index
  right_y = amp
  loop_range = range(mean_index, -1,
                     -1) if to_left else range(mean_index, len(x_set))
  for i in loop_range:
    if y_set[i] < amp / 2.:
      if to_left:
        left_x = x_set[i]
        left_y = y_set[i]
        right_x = x_set[i + 1]
        right_y = y_set[i + 1]
      else:
        left_x = x_set[i - 1]
        left_y = y_set[i - 1]
        right_x = x_set[i]
        right_y = y_set[i]
      break
  index = find_x_of_point_on_line(left_x, left_y, right_x, right_y, amp / 2.)
  return index


def parabola(x, a, b, c):
  return a * x * x + b * x + c


def find_x_of_point_on_line(x1, y1, x2, y2, y_value):
  a = (y1 - y2) / (x1 - x2)
  b = y1 - a * x1
  x_value = (y_value - b) / a
  return x_value


def find_mean(x_set, y_set):
  n = len(x_set)
  assert n != 0
  assert len(x_set) <= len(y_set)

  amp = y_set[0]
  mean = x_set[0]
  mean_index = 0
  for i in range(n - 1):
    if y_set[i] > amp:
      amp = y_set[i]
      mean = x_set[i]
      mean_index = i
  return mean, mean_index, amp


def get_fwhm_nema(x_set, y_set):
  _, mean_index, _ = find_mean(x_set, y_set)
  x_data_for_fit = [
      x_set[mean_index - 1], x_set[mean_index], x_set[mean_index + 1]
  ]
  y_data_for_fit = [
      y_set[mean_index - 1], y_set[mean_index], y_set[mean_index + 1]
  ]
  #disable false-positive warning
  #pylint: disable=unbalanced-tuple-unpacking
  popt, _ = curve_fit(f=parabola, xdata=x_data_for_fit, ydata=y_data_for_fit)
  a = popt[0]
  b = popt[1]
  c = popt[2]
  max_value = -(b * b - 4 * a * c) / (4 * a)
  right = find_half(x_set, y_set, max_value, to_left=False)
  left = find_half(x_set, y_set, max_value, to_left=True)
  fwhm = right - left
  return fwhm


def get_fwhm_approximation(x_set, y_set):
  right = find_half(x_set, y_set, to_left=False)
  left = find_half(x_set, y_set, to_left=True)
  fwhm = right - left
  return fwhm


def print_fwhm(x, y):
  fwhm_approximation = get_fwhm_approximation(x, y)
  print(
      f'FWHM value from linear approximation technique: {fwhm_approximation}'
  )
  fwhm_nema = get_fwhm_nema(x, y)
  print(f'FWHM value from NEMA technique: {fwhm_nema }')


def main(args):

  out_files = get_out_files(args.out_file)

  # the third argument is the matrix_size for a moment not used.
  image, voxel_size, _ = load_image_and_metadata(args.header)

  if args.normalize:
    image /= np.sum(image)

  max_voxels_x, max_voxels_y, max_voxels_z = find_voxels_with_maximum_values(
      image
  )
  max_voxels_x = max_voxels_x[0]
  max_voxels_y = max_voxels_y[0]
  max_voxels_z = max_voxels_z[0]
  # if user provided position
  max_voxels_x = get_bin_by_position(
      args.set_x, voxel_size[0], len(image[:, 0, 0])
  ) if args.set_x is not sys.maxsize else max_voxels_x
  max_voxels_y = get_bin_by_position(
      args.set_y, voxel_size[1], len(image[0, :, 0])
  ) if args.set_y is not sys.maxsize else max_voxels_y
  max_voxels_z = get_bin_by_position(
      args.set_z, voxel_size[2], len(image[0, 0, :])
  ) if args.set_z is not sys.maxsize else max_voxels_z
  # profile with one y voxel and one z voxel
  x_profile = get_profile_x_3d(
      image, max_voxels_y - args.around_bins, max_voxels_y + args.around_bins,
      max_voxels_z - args.around_bins, max_voxels_z + args.around_bins
  )
  y_profile = get_profile_y_3d(
      image, max_voxels_x - args.around_bins, max_voxels_x + args.around_bins,
      max_voxels_z - args.around_bins, max_voxels_z + args.around_bins
  )
  z_profile = get_profile_z_3d(
      image, max_voxels_x - args.around_bins, max_voxels_x + args.around_bins,
      max_voxels_y - args.around_bins, max_voxels_y + args.around_bins
  )

  plot_1d_profile(
      x_profile, voxel_size[0], "cm", "x", out_files[0], args.disable_display
  )
  plot_1d_profile(
      y_profile, voxel_size[1], "cm", "y", out_files[1], args.disable_display
  )
  plot_1d_profile(
      z_profile, voxel_size[2], "cm", "z", out_files[2], args.disable_display
  )

  # 2D profiles
  xy_profile = get_profile_xy(
      image, max_voxels_z - args.around_bins, max_voxels_z + args.around_bins
  )
  xz_profile = get_profile_xz(
      image, max_voxels_y - args.around_bins, max_voxels_y + args.around_bins
  )
  yz_profile = get_profile_yz(
      image, max_voxels_x - args.around_bins, max_voxels_x + args.around_bins
  )
  plot_2d_profile(
      xy_profile, [voxel_size[0], voxel_size[1]],
      units="cm",
      title=["x", "y"],
      out_file=out_files[3],
      disable_display=args.disable_display
  )
  plot_2d_profile(
      xz_profile, [voxel_size[0], voxel_size[2]],
      units="cm",
      title=["x", "z"],
      out_file=out_files[4],
      disable_display=args.disable_display
  )
  plot_2d_profile(
      yz_profile, [voxel_size[1], voxel_size[2]],
      units="cm",
      title=["y", "z"],
      out_file=out_files[5],
      disable_display=args.disable_display
  )


if __name__ == "__main__":
  EXAMPLE_USAGE_DESCRIPTION = '''
  Image_tools can be used to generate peak profiles from the image in the interfile format.
  The profiles are done for the maximum intensity bins found in the image matrix.
  If the out_file argument is given then the profile plots are saved in the png format.
  Otherwise the profiles are only plotted.
  Example usage:

  ./image_tools.py --header dabc_21026133329_CASTOR_it20.hdr
  ./image_tools.py --header dabc_21026133329_CASTOR_it20.hdr --out_file myProfiles
  '''
  PARSER = argparse.ArgumentParser(
      epilog=EXAMPLE_USAGE_DESCRIPTION,
      formatter_class=argparse.RawDescriptionHelpFormatter
  )
  PARSER.add_argument(
      "--header", dest="header", help="image header file", required=True
  )
  PARSER.add_argument(
      "--disable_display",
      help="""this argument disables display of profiles,
      it should be used together with \'--out_file\' to save results to file""",
      required=False,
      action='store_true'
  )
  PARSER.add_argument(
      "--normalize",
      help="normalize to one",
      required=False,
      action='store_true'
  )
  PARSER.add_argument(
      "--out_file",
      help=
      "output file base name that will be used to write output figures in png format ",
      required=False,
      dest='out_file'
  )
  PARSER.add_argument(
      "--around_bins",
      help="number of bins around max value that will be used in plot",
      type=int,
      required=False,
      default=1,
  )
  PARSER.add_argument(
      "--set_x",
      help=
      "use provided X value to generate profile, instead of finding index of max value in X, in cm, assuming center of the image is in 0",
      type=int,
      required=False,
      default=sys.maxsize,
  )
  PARSER.add_argument(
      "--set_y",
      help=
      "use provided Y value to generate profile, instead of finding index of max value in Y, in cm, assuming center of the image is in 0",
      type=int,
      required=False,
      default=sys.maxsize,
  )
  PARSER.add_argument(
      "--set_z",
      help=
      "use provided Z value to generate profile, instead of finding index of max value in Z, in cm, assuming center of the image is in 0",
      type=int,
      required=False,
      default=sys.maxsize,
  )
  main(PARSER.parse_args())
