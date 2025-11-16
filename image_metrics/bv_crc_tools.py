"""

Tools for calculation of bv/crc for NEMA and XCAT phantom

"""
import itertools

import numpy as np
from scipy.stats import chi2

from image_metrics.interfile_tools import get_image
from image_metrics.interfile_tools import get_interfiles
from image_metrics.interfile_tools import interfile2array
from image_metrics.interfile_tools import interfile_parser
from image_metrics.interfile_tools import read_parameters
from image_tools.gate_macro_parser.phantom_parser import JPET
from image_tools.gate_macro_parser.phantom_parser import get_translation

# center of background rois mm
BKG_CENTERS = [
    (111.5105, -1.270829), (72.63656, 56.083368), (7.133568e-15, 81.5),
    (-67.88933, 59.674647), (-94.70335, 32.849289), (-111.5105, -1.270829),
    (-116.31280000000001, -39.168228), (-97.61264, -72.413797),
    (115.48389999999999, -44.667894), (86.48913, -78.478255), (-61.0, -81.5),
    (-2.5, -81.5)
]


def convert_circle_to_mask(position, radius, scale_factors, sizes):
  """
  Convert circle (defined in XY plane) to image mask, represented as 3D array with following values:
    * True - for voxels inside circle
    * False - for voxels outside circe

  Parameters
  ----------
  position : tuple
      Circle center defined as tuple of x0, y0, z0 values
  radius : float
      Circle radius
  scale_factors : tuple
      Tuple of voxels sizes in mm along X,Y,Z
  sizes : tuple
      Image size in number of voxels along X,Y,Z

  Returns
  -------
  np.ndarray
      Mask
  """
  x0, y0, z0 = position
  scaling_x, scaling_y, scaling_z = scale_factors
  size_x, size_y, size_z = sizes

  def mm2pix(dm, dx, size):
    return int(size / 2 + (dm * (1 / dx)))

  x0 = mm2pix(x0, scaling_x, size_x)
  y0 = mm2pix(-y0, scaling_y, size_y)
  z0 = mm2pix(z0, scaling_z, size_z)

  radius_x = int(radius * (1 / scaling_x))
  radius_y = int(radius * (1 / scaling_y))

  y, x = np.ogrid[-radius_y:radius_y + 1, -radius_x:radius_x + 1]
  index = x**2 / radius_x**2 + y**2 / radius_y**2 <= 1.
  test_array = np.full(
      (size_z, size_y, size_x), fill_value=np.ma.nomask, dtype=np.ma.MaskType
  )
  test_array[z0, y0 - radius_y:y0 + radius_y + 1,
             x0 - radius_x:x0 + radius_x + 1][index] = True

  return test_array


def calculate_roi_stats(image, roi_masks):
  """
  Calculate statistics of ROIs

  Parameters
  ----------
  image : np.ndarray
      Image
  roi_masks : typing.Iterable[np.ndarray]
      Collection of masks to apply on image

  Returns
  -------
  tuple[float, float, float, float]
      Mean, standard deviation, variance and size of ROI
  """
  roi_sum = 0.
  roi_sum_sq = 0.
  roi_size = 0

  for roi_mask in roi_masks:
    roi_data = image[roi_mask]
    roi_sum += np.sum(roi_data)
    roi_sum_sq += np.sum(np.square(roi_data))
    roi_size += roi_data.size

  roi_mean = roi_sum / roi_size
  roi_var = roi_sum_sq / roi_size - roi_mean * roi_mean
  roi_std = np.sqrt(roi_var)

  return roi_mean, roi_std, roi_var, roi_size


def get_bv_confidence_intervals_naive(bv_val, n_tot, alpha):
  """
  Get confidence intervals

  Naive approach

  Parameters
  ----------
  bv_val : float
  n_tot : float
  alpha : float

  Returns
  -------
  tuple[float, float]
  """

  def term(n, u):
    return np.sqrt((n - 1) / u)

  u1 = chi2.ppf(1 - alpha / 2, n_tot - 1)
  u2 = chi2.ppf(alpha / 2, n_tot - 1)
  lcl = bv_val * term(n_tot, u1)
  ucl = bv_val * term(n_tot, u2)
  return lcl, ucl


def calculate_bv_value(background_std, background_mean):
  """
  Calculate BV value

  Parameters
  ----------
  background_std : float
  background_mean : float

  Returns
  -------
  float
  """
  return background_std / background_mean


def calculate_bv_error(
    bv_val, n_tot, get_ci=get_bv_confidence_intervals_naive
):
  """
  Calculate BV error

  In principle it is correct only for naive approach

  Parameters
  ----------
  bv_val : float
  n_tot : float
  get_ci : function

  Returns
  -------
  float
  """
  factor = 1.96
  alpha = 0.05
  _, ucl = get_ci(bv_val, n_tot, alpha)
  return (ucl - bv_val) / factor


def calculate_bv(image, bv_roi_masks):
  """
  Calculate BV metrics

  Parameters
  ----------
  image : np.ndarray
      Image
  bv_roi_masks : typing.Iterable[np.ndarray]
      Collection of masks to apply on image

  Returns
  -------
  tuple[float, float, float, float, float]
      BV value, BV error, BV ROI mean, BV ROI variance, BV ROI size
  """
  roi_mean, roi_std, roi_var, roi_size = calculate_roi_stats(
      image, bv_roi_masks
  )

  bv_value = calculate_bv_value(roi_std, roi_mean)
  bv_error = calculate_bv_error(bv_value, roi_size)

  return bv_value, bv_error, roi_mean, roi_var, roi_size


def calculate_crc_value(
    roi_mean, background_mean, roi_activity, background_activity
):
  """
  Calculate CRC value

  Parameters
  ----------
  roi_mean : float
  background_mean : float
  roi_activity : float
  background_activity : float

  Returns
  -------
  float
  """
  return (roi_mean / background_mean - 1.) / (
      roi_activity / background_activity - 1.
  )


def calculate_crc_error(
    roi_mean, roi_var, roi_size, bv_mean, bv_var, bv_roi_size, roi_activity,
    background_activity
):
  """
  Calculate CRC error

  Parameters
  ----------
  roi_mean : float
  roi_var : float
  roi_size : float
  bv_mean : float
  bv_var : float
  bv_roi_size : float
  roi_activity : float
  background_activity : float

  Returns
  -------
  float
  """
  return np.sqrt(
      roi_var / roi_size + (roi_mean * roi_mean * bv_var) /
      (bv_roi_size * bv_mean * bv_mean)
  ) / ((roi_activity / background_activity - 1.) * bv_mean)


def calculate_crc(
    image, roi_masks, activity_hot, activity_bkg, bv_mean, bv_var, bv_roi_size
):
  """
  Calculate CRC metrics

  Parameters
  ----------
  image : np.ndarray
      Image
  roi_masks : typing.Iterable[np.ndarray]
      Collection of masks to apply on image
  activity_hot : float
      True activity of the hot region
  activity_bkg : float
      True activity of background regions
  bv_mean : float
      BV ROI mean
  bv_var : float
      BV ROI variance
  bv_roi_size : float
      BV ROI size

  Returns
  -------
  tuple[float, float, float, float, float]
      CRC value, CRC error, CRC ROI mean, CRC ROI variance, CRC ROI size
  """
  roi_mean, _, roi_var, roi_size = calculate_roi_stats(image, roi_masks)

  crc_value = calculate_crc_value(
      roi_mean, bv_mean, activity_hot, activity_bkg
  )
  crc_error = calculate_crc_error(
      roi_mean, roi_var, roi_size, bv_mean, bv_var, bv_roi_size, activity_hot,
      activity_bkg
  )

  return crc_value, crc_error, roi_mean, roi_var, roi_size


def calculate_bv_crc(
    image, bv_roi_masks, crc_roi_masks, activity_hot, activity_background
):
  """
  Calculate BV and CRC metrics

  Parameters
  ----------
  image : np.ndarray
      Image
  bv_roi_masks : typing.Iterable[np.ndarray]
      Collection of background masks to apply on image
  crc_roi_masks : typing.Iterable[np.ndarray]
      Collection of hot region masks to apply on image
  activity_hot : float
      True activity of the hot region
  activity_background : float
      True activity of background regions

  Returns
  -------
  tuple[float, float, float, float]
      BV value, BV error, CRC value, CRC error
  """
  bv_value, bv_delta, bv_mean, bv_var, bv_roi_size = calculate_bv(
      image=image, bv_roi_masks=bv_roi_masks
  )
  crc_value, crc_delta, _, _, _ = calculate_crc(
      image=image,
      roi_masks=crc_roi_masks,
      activity_hot=activity_hot,
      activity_bkg=activity_background,
      bv_mean=bv_mean,
      bv_var=bv_var,
      bv_roi_size=bv_roi_size
  )
  return bv_value, bv_delta, crc_value, crc_delta


def get_jpet_r_x_y_z(volume_name):
  """

  Parameters
  ----------
  volume_name : str
      Volume name
  Returns
  -------
  tuple[float, float, float, float]
      Volume radius and position in mm
  """
  r = JPET.geometry[volume_name]['setRmax'] * 10
  x, y, z = get_translation(JPET.geometry, volume_name)
  return r, x, y, z


def get_bv_crc_nema(
    interfile_header,
    volume_name,
    activity_hot=4,
    activity_background=1,
    bkg_centers=None,
    z_slices=None,
    **kwargs
):
  """
  Calculate BV and CRC metrics using named volumes as region masks

  Parameters
  ----------
  interfile_header : str
      Path to interfile (.hdr) of image to analyze
  volume_name : str
      Name of volume to calculate masks
  activity_hot : float | optional
      True activity of the hot region
  activity_background : float | optional
      True activity of background regions

  Other Parameters
  ----------------
  bkg_centers : list[tuple[float, float]] | None
      List of background volume centers in XY plane, if not provided default values for NEMA phantom
      will be used
  z_slices : list[float] | None
      List of z slices, if not provided default values for NEMA phantom will be used
  **kwargs :
      Additional parameters for image_metrics.interfile_tools.get_image function, smoothing
      parameters: `sigma_smoothing`

  Examples
  --------
  Call with smoothing parameter:

    `get_bv_crc_nema(interfile_header=hdr_path, volume_name="sphere10in", activity_hot=5.5,
    activity_background=0.5, sigma_smoothing=1)`

  Returns
  -------
  tuple[float, float, float, float]
      BV value, BV error, CRC value, CRC error
  """
  if bkg_centers is None:
    bkg_centers = BKG_CENTERS

  if z_slices is None:
    z_slices = [-24, -12, 0, 12, 24]

  image = get_image(interfile_header, kwargs)
  parameters = interfile_parser(interfile_header)

  r, x, y, z = get_jpet_r_x_y_z(volume_name)
  scaling_factor_xy, scaling_factor_z, size_xy, size_z = read_parameters(
      parameters
  )

  def bv_mask_generator():
    for delta_z, bkg_center in itertools.product(z_slices, bkg_centers):
      yield convert_circle_to_mask(
          position=(bkg_center[0], bkg_center[1], z + delta_z),
          radius=r,
          scale_factors=(
              scaling_factor_xy, scaling_factor_xy, scaling_factor_z
          ),
          sizes=(size_xy, size_xy, size_z)
      )

  def crc_mask_generator():
    yield convert_circle_to_mask(
        position=(x, y, z),
        radius=r,
        scale_factors=(scaling_factor_xy, scaling_factor_xy, scaling_factor_z),
        sizes=(size_xy, size_xy, size_z)
    )

  return calculate_bv_crc(
      image, bv_mask_generator(), crc_mask_generator(), activity_hot,
      activity_background
  )


def get_bv_crc_xcat(
    interfile_header, bv_roi_files, crc_file, activity_hot,
    activity_background, **kwargs
):
  """
  Calculate BV and CRC metrics using images as region masks

  Parameters
  ----------
  interfile_header : str
      Path to interfile (.hdr) of image to analyze
  bv_roi_files : list[str]
      Collection of paths to background mask files
  crc_file : str
      Path to hot region mask file
  activity_hot : float
      True activity of the hot region
  activity_background : float
      True activity of background regions

  Other Parameters
  ----------------
  **kwargs :
      Additional parameters for image_metrics.interfile_tools.get_image function, smoothing
      parameters: `sigma_smoothing`

  Returns
  -------
  tuple[float, float, float, float]
      BV value, BV error, CRC value, CRC error
  """
  image: np.ndarray = get_image(interfile_header, kwargs)
  parameters = interfile_parser(interfile_header)

  def mask_generator(roi_files):
    for roi_file in roi_files:
      yield interfile2array({**parameters, **{'path_to_data_file': roi_file}}) \
        .astype(np.ma.MaskType)

  return calculate_bv_crc(
      image, mask_generator(bv_roi_files), mask_generator([crc_file]),
      activity_hot, activity_background
  )


def get_bv_crc_from_dir(interfile_path, volume_name, return_fstrings=True):
  """
  Calculate image quality CRC BV
  """
  table_data = [['Filename', 'BV', 'BV error', 'CRC', 'CRC error']]
  file_paths = get_interfiles(interfile_path)

  for interfile_header in file_paths:

    bv_value, bv_delta, crc_value, crc_delta = get_bv_crc_nema(
        interfile_header, volume_name
    )
    if return_fstrings:
      table_data.append(
          [
              interfile_header, f"{bv_value}.3f", f"{bv_delta}.3f",
              f"{crc_value}.2f", f"{crc_delta}.2f"
          ]
      )
    else:
      table_data.append(
          [interfile_header, bv_value, bv_delta, crc_value, crc_delta]
      )
  return table_data
