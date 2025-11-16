"""
Calculation of BV and CRC metrics toghether with its confidence intervals and
standard errors.
"""

import numpy as np
from scipy.stats import chi2


def get_bv(voxels):
  return np.std(voxels) / np.mean(voxels)


def get_bv_confidence_intervals_naive(voxels, alpha):
  '''
  Naive CI determination assuming purely Gaussian distribution.
  '''

  def term(n, u):
    return np.sqrt((n - 1) / u)

  bv_val = get_bv(voxels)
  n_tot = len(voxels)
  u1 = chi2.ppf(1 - alpha / 2, n_tot - 1)
  u2 = chi2.ppf(alpha / 2, n_tot - 1)
  lcl = bv_val * term(n_tot, u1)
  ucl = bv_val * term(n_tot, u2)
  return lcl, ucl


def get_bv_confidence_intervals2(voxels, alpha):
  ''' McKay correction
    McKay, A. T. (1932)
    Distributions of the coefficient of variation and the extended ‘t’ distribution.
    Journal of the Royal Statistical Society, Vol. 95, pp. 695-698.
  '''

  def denom(bv_val, n, u):
    return np.sqrt(bv_val * bv_val * (u / n - 1.0) + (u / (n - 1)))

  bv_val = get_bv(voxels)
  n_tot = len(voxels)
  u1 = chi2.ppf(1 - alpha / 2, n_tot - 1)
  u2 = chi2.ppf(alpha / 2, n_tot - 1)
  lcl = bv_val * 1 / denom(bv_val, n_tot, u1)
  ucl = bv_val * 1 / denom(bv_val, n_tot, u2)
  return lcl, ucl


def get_bv_confidence_intervals3(voxels, alpha):
  ''' Vangel's correction
    Vangel, M. (1996)
    Confidence intervals for a normal coefficient of variation.
    American Statistician, Vol. 15, No. 1, pp. 21-26
  '''

  def denom(bv_val, n, u):
    return np.sqrt(bv_val * bv_val * ((u + 2) / n - 1.0) + (u / (n - 1)))

  bv_val = get_bv(voxels)
  n_tot = len(voxels)
  u1 = chi2.ppf(1 - alpha / 2, n_tot - 1)
  u2 = chi2.ppf(alpha / 2, n_tot - 1)
  lcl = bv_val * 1 / denom(bv_val, n_tot, u1)
  ucl = bv_val * 1 / denom(bv_val, n_tot, u2)
  return lcl, ucl


def get_bv_std(voxels, get_ci=get_bv_confidence_intervals_naive):
  ''' In principle it is correct only for naive approach
  '''
  factor = 1.96
  alpha = 0.05
  bv_val = get_bv(voxels)
  _, ucl = get_ci(voxels, alpha)
  return (ucl - bv_val) / factor
