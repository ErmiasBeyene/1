""" This module provides some functions to calculate the metrics corresponding
    to images similarity. These functions are:
 - correlation functions (giving simple correlation value),
 - standard deviation functions,
 - mean squared error functions (mse),
 - structural similarity index measure (ssim)
 - peak signal to noise function (pnsr).
 Described functions can be found here:
 [1] Renieblas, Gabriel Prieto et al. “Structural similarity index family for image quality assessment in radiological
 images.” Journal of medical imaging (Bellingham, Wash.) vol. 4,3 (2017): 035501. doi:10.1117/1.JMI.4.3.035501
 [2] Tanabe Y, Ishida T. Quantification of the accuracy limits of image registration using peak signal-to-noise ratio.
 Radiol Phys Technol. 2017 Mar;10(1):91-94. doi: 10.1007/s12194-016-0372-3. Epub 2016 Aug 18. PMID: 27539271.
"""

import argparse
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def flatten_image(img1, img2=np.array(0)):
  """
  Function generates flattened, one-dimensional versions of given images.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array (optional)
   second image

  Returns
  -------
  flattened one-dimensional version of 1st and 2nd image : list,list
  """
  flattened_img1 = img1.flatten().tolist()
  flattened_img2 = img2.flatten().tolist()
  return flattened_img1, flattened_img2


def shape_function(img1, dim):
  """
  Function calculate metrics associated with image's dimensionality.

  Parameters
  ----------
  img : np.array
   image
  dim : int
   dimensionality of the input file

  Returns
  -------
  relative dimensionality,shape in x-,y-,z-direction : int,int,int,int
  """
  sh_x, sh_y, sh_z = img1.shape
  m = int((dim - 1) / 2)
  return m, sh_x, sh_y, sh_z


def correlation(img1, img2):
  """
  Function calculating simple correlation between two images.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image

  Returns
  -------
  correlation coefficient : float
  """
  flattened_img1, flattened_img2 = flatten_image(img1, img2)
  cov = np.cov(flattened_img1, flattened_img2, ddof=1)
  return cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]))


def mcorrelation_arithmetic(img1, img2, dim):
  """
  Function calculating arithmetic correlation between two images.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image
  dim : int
   dimensionality of the input file

  Returns
  -------
  arithmetic correlation coefficient : float
  """
  m, sh_x, sh_y, sh_z = shape_function(img1, dim)
  mcor = 0.0
  size = (sh_x - 2 * m) * (sh_y - 2 * m) * (sh_z - 2 * m)
  for x in range(m, sh_x - m):
    for y in range(m, sh_y - m):
      for z in range(m, sh_z - m):
        mcor += correlation(
            img1[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1],
            img2[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1],
        )
  mcor /= size
  return mcor


def mcorrelation_multiplicative(img1, img2, dim):
  """
  Function calculating multiplicative correlation between two images.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image
  dim : int
   dimensionality of the input file

  Returns
  -------
  multiplicative correlation coefficient : float
  """
  m, sh_x, sh_y, sh_z = shape_function(img1, dim)
  mcor = 1.0
  size = (sh_x - 2 * m) * (sh_y - 2 * m) * (sh_z - 2 * m)
  for x in range(m, sh_x - m):
    for y in range(m, sh_y - m):
      for z in range(m, sh_z - m):
        mcor *= correlation(
            img1[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1],
            img2[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1],
        )
  mcor = mcor**(1.0 / float(size))
  return mcor


def standard_dev(img):
  """
  Function calculating standard deviation, across one image.

  Parameters
  ----------
  img : np.array
   image

  Returns
  --------
  standard deviation coefficient : float
  """
  flatten_image(img)
  return np.std(img)


def standard_dev_mask(img1, img2, dim):
  """
  Function calculating standard deviation with a mask applied on the image (taking into account some parts of the arrays).

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image
  dim : int
   dimensionality of the input file

  Returns
  -------
  correlation between masked arrays : float
  """
  m, sh_x, sh_y, sh_z = shape_function(img1, dim)
  std_matrix1 = np.empty([1])
  std_matrix2 = np.empty([1])
  for x in range(m, sh_x - m):
    for y in range(m, sh_y - m):
      for z in range(m, sh_z - m):
        std_matrix1 = np.append(
            std_matrix1,
            standard_dev(
                img1[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1]
            ),
        )
        std_matrix2 = np.append(
            std_matrix2,
            standard_dev(
                img2[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1]
            ),
        )
  result = correlation(std_matrix1, std_matrix2)
  return result


def my_ssim(img1, img2, img_dynamic_range, alpha, beta, gamma):
  """
  Szymon's version of calculating ssim (structural similarity index).

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image
  img_dynamic_range : int
   dynamic range of image (the range of tonal difference between the lightest light and darkest dark of an image)
  alpha : float
   structural similarity weight
  beta : float
   structural similarity weight
  gamma : float
   structural similarity weight

  Returns
  -------
  Szymon's SSIM : float
  """
  flattened_img1, flattened_img2 = flatten_image(img1, img2)

  mean1 = np.mean(flattened_img1)
  mean2 = np.mean(flattened_img2)

  cov = np.cov(flattened_img1, flattened_img2, ddof=1)

  std1 = np.sqrt(cov[0, 0])
  std2 = np.sqrt(cov[1, 1])

  cov12 = cov[0, 1]

  _k1 = 0.01
  _k2 = 0.03
  c1 = (_k1 * img_dynamic_range)**2
  c2 = (_k2 * img_dynamic_range)**2
  c3 = c2 / 2.0

  luminance = (2.0 * mean1 * mean2 + c1) / (mean1**2 + mean2**2 + c1)
  contrast = (2.0 * std1 * std2 + c2) / (std1**2 + std2**2 + c2)
  structure = (cov12 + c3) / (std1 * std2 + c3)

  return luminance**alpha * contrast**beta * structure**gamma


def mssim(img1, img2, data_range, dim, alpha, beta, gamma):
  """
  This function calculates mean structural similarity index (MSSIM) for quantitativeanalysis of image.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image
  data_range : int
   data range of image
  dim : int
   dimensionality of the input file
  alpha : float
   structural similarity weight
  beta : float
   structural similarity weight
  gamma : float
   structural similarity weight

  Returns
  -------
  MSSIM : float
  """
  sh_x, sh_y, sh_z = img1.shape
  m = int((dim - 1) / 2)
  mssim_ = 0.0
  size = (sh_x - 2 * m) * (sh_y - 2 * m) * (sh_z - 2 * m)
  for x in range(m, sh_x - m):
    for y in range(m, sh_y - m):
      for z in range(m, sh_z - m):
        mssim_ += my_ssim(
            img1[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1],
            img2[x - m:x + m + 1, y - m:y + m + 1,
                 z - m:z + m + 1], data_range, alpha, beta, gamma
        )
  mssim_ /= size
  return mssim_


def mse(img1, img2):
  """
  Function calculating mean squared error (MSE) between two images.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image

  Returns
  -------
  MSE : float
  """
  return np.mean((img1 - img2)**2.0)


def mmse(img1, img2, dim):
  """
  Function calculating masked mean squared error (MMSE) between two images.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image
  dim : int
   dimensionality of the input file

  Returns
  -------
  MMSE : float
  """
  m, sh_x, sh_y, sh_z = shape_function(img1, dim)
  mmse_ = 0.0
  size = (sh_x - 2 * m) * (sh_y - 2 * m) * (sh_z - 2 * m)
  for x in range(m, sh_x - m):
    for y in range(m, sh_y - m):
      for z in range(m, sh_z - m):
        mmse_ += mse(
            img1[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1],
            img2[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1],
        )
  mmse_ /= size
  return mmse_


def rmse(img1, img2):
  """
  Function calculating root mean squared error (RMSE) between two images.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image

  Returns
  -------
  RMSE : float
  """
  return np.sqrt(mse(img1, img2))


def my_psnr(img1, img2, image_range_):
  """
  Szymon's version of peak to noise ratio (PSNR) function.

  Parameters
  ----------
  img1 : np.array
   first image
  img2 : np.array
   second image
  image_range_ : int
   image range

  Returns
  -------
  PSNR : float
  """
  return 10.0 * np.log10((image_range_**2) / mse(img1, img2))


def main():
  parser = argparse.ArgumentParser(
      description=
      "Process two images and calculate some metrics to compare them."
  )
  parser.add_argument(
      "--img1_path", help="Path to the first image", metavar="FILE"
  )
  parser.add_argument(
      "--img2_path", help="Path to the second image", metavar="FILE"
  )
  parser.add_argument("--number_format", help="Set number format e.g. float64")
  parser.add_argument(
      "--mat_x", help="First dimension of reshaped matrix", type=int
  )
  parser.add_argument(
      "--mat_y", help="Second dimension of reshaped matrix", type=int
  )
  parser.add_argument(
      "--mat_z", help="Third dimension of reshaped matrix", type=int
  )
  parser.add_argument(
      "--dim", help="Dimensionality of the input file", type=int
  )
  parser.add_argument("--alpha", help="Structural similarity weight")
  parser.add_argument("--beta", help="Structural similarity weight")
  parser.add_argument("--gamma", help="Structural similarity weight")
  args = parser.parse_args()

  image1 = np.fromfile(
      args.img1_path, dtype=args.number_format
  ).reshape((args.mat_x, args.mat_y, args.mat_z), order="F")
  image2 = np.fromfile(
      args.img2_path, dtype=args.number_format
  ).reshape((args.mat_x, args.mat_y, args.mat_z), order="F")

  image1 = image1.astype(np.float64)
  image2 = image2.astype(np.float64)
  image_range = max(image2.flatten().tolist()) - min(image2.flatten().tolist())

  print("Szymon's implementation")
  print(f"Data range length: {image_range}")
  print(f"MMSE    : {mmse(image1, image2,args.dim)}")
  print(f"RMSE    : {rmse(image1, image2)}")
  print(f"PSNR    : {my_psnr(image1, image2, image_range)}")
  my_ssim_value = my_ssim(
      image1, image2, image_range, args.alpha, args.beta, args.gamma
  )
  print(f"SSIM    : {my_ssim_value}")
  mssim_value = mssim(
      image1, image2, image_range, args.dim, args.alpha, args.beta, args.gamma
  )
  print(f"MSSIM   : {mssim_value}")
  print(f"Std     : {standard_dev_mask(image1, image2,args.dim)}")
  mcorr_aryt = round(mcorrelation_arithmetic(image1, image2, args.dim), 3)
  mcorr_mult = round(mcorrelation_multiplicative(image1, image2, args.dim), 3)
  print(f"MCorrelationArythmetic     : {mcorr_aryt}")
  print(f"MCorrelationMultiplicative : {mcorr_mult}")
  print("From skimage package")
  mse_value = mean_squared_error(image1, image2)
  ssim_value = ssim(image1, image2, data_range=image_range, win_size=3)
  psnr_value = psnr(image1, image2, data_range=image_range)
  print(f"MSE     : {mse_value}")
  print(f"MSSIM   : {ssim_value}")
  print(f"PSNR    : {psnr_value}")


if __name__ == "__main__":
  main()
