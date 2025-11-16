"""
BV/CRC analyzer

Produces BV/CRC plots for given

To execute for single dictionary:
python bv_crc_analyzer.py -idir /path/1/to/dir -lc="blue" -ls="-." -ld="description of line"

To execute for many directories (here for 2):
python bv_crc_analyzer.py -idir /path/1/to/dir -lc="blue" -ls="-." -ld="description of line" -idir /path/2/to/dir
-lc="blue" -ls=".." -ld="description of line 2"

where:
-idir is a pth to directory with *.hdr files
-lc is a line color (this can be a word like "blue" or hex representation of color like "#CD5C5C")
-ls is a line style (this command does not work with "--" but other options are fine)
-ld is a description of line (this decription will be displayed on the legend of plot)

For more examples of possible values  for  -ls and -lc options, check matplot line styles and colors.
"""
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from image_metrics.bv_crc_tools import get_bv_crc_from_dir


def fill_results_for_files(spheres, source_dir_path, index, results):
  """
  Calculate & fill results for given files.

  Parameters
  ----------
  spheres : list
   list of phantom spheres
  source_dir_path : Path
  index : int
   directory's (with *.hdr files) index e.g. 1st added directory has index equals 0 and so on
  reults : np.array
   an array with calculated values of BV and CRC
  """
  data_dir_path = "".join([os.fspath(source_dir_path), "/"])
  for sphere_index, sphere_name in enumerate(spheres):
    table_data = get_bv_crc_from_dir(
        data_dir_path, sphere_name, return_fstrings=False
    )[1:]  #we omits 1st element of list because it contains titles of columns
    for file_index, data in enumerate(table_data):
      results[sphere_index, file_index, :, index] = data[1:]
      #we omits 1st element of list because it contains a path to *.hdr file which was used to calculate BV and CRC


def get_file_title(file_path):
  """
  Extract a title of file.
  e.g. a title of filetitle1.hdr is filetitle

  Parameters
  ----------
  file_path : Path
   file path
  Returns
  -------
  a title of file : str
  """
  for i, char in reversed(list(enumerate(file_path.stem))):
    if not char.isnumeric():
      return file_path.stem[0:i + 1]
  return file_path.stem


def get_file_index(file_path, file_title):
  """
  Returns an index of given file.
  e.g. an index of filetitle1.hdr is 1

  Parameters
  ----------
  file_path: Path
   path to file
  file_title : str
   e.g. a title of filetitle1.hdr is filetitle

  Returns
  ------
  an index of file: int
  """
  return int(file_path.stem.split(file_title)[1])


def get_hdr_files_paths(directory_path, from_nth=None, to_nth=None):
  """
  For given directory method returns a list of paths to files.
  Files in the list are in the order - first iteration file is at the beginning, and so on.

  Parameters
  ----------
  directory_path : Path
   path to the directory with *.hdr files
  from_nth : int (optional, default is None)
   from which iteration starts analysis
  to_nth : int (optional, default is None)
   to which iteration starts analysis

  Returns
  -------
  paths to the *.hrd files in the order : list
  """
  files_paths = list(directory_path.glob("*.hdr"))
  file_title = get_file_title(files_paths[0])
  files_paths = sorted(
      files_paths, key=lambda fp: get_file_index(fp, file_title)
  )

  from_index = get_file_index(
      files_paths[0], file_title
  ) if from_nth is None else from_nth

  to_index = get_file_index(
      files_paths[-1], file_title
  ) if to_nth is None else to_nth

  return [
      fp for fp in files_paths
      if from_index <= get_file_index(fp, file_title) <= to_index
  ]


def generate_results(directories_paths, from_nth, to_nth):
  """
  From given directories with data files it produces array with calculated BV and CRC.

  Parameters
  ----------
  directories_paths : list
   patch to directories with *.hdr files, each entr is Path object
  from_nth : int (optional, default is None)
   from which iteration starts analysis
  to_nth : int (optional, default is None)
   to which iteration starts analysis

  Returns
  -------
  calculated values of BV and CRC : np.array
   format of array is: [sphere_index, iteration, values of calculated BV and CRC, legend index (so for which directory)]
  """
  spheres = ["sphere10in", "sphere13in", "sphere17in", "sphere22in"]
  files_paths = [
      get_hdr_files_paths(dp, from_nth, to_nth) for dp in directories_paths
  ]
  iterations = min([len(fp) for fp in files_paths])
  #Output of method measure_single_file from stat_interfile returns list with 5 arguments but first
  #argument is a file path, hence we dont need it - that is why below 3rd argument is 4
  results = np.zeros((len(spheres), iterations, 4, len(directories_paths)))
  for index, dir_path in enumerate(directories_paths):
    fill_results_for_files(spheres, dir_path, index, results)
  return results, spheres


def plot_results(results, spheres, legend, line_colors, line_styles):
  """
  Method plots given results in a form of 4 subplots.
  At the end of work methods shows window with an image and user has to decide what to do next.

  Parameters
  ----------
  results : np.array
   calculated results [sphere index, iteration, BV and CRC values, directory index (== legend index)]
  spheres : lits
   list of spheres for which was calculated values of BC and CRC
  legend : list
   descriptions of directories e.g. ["description A", "description B", "description C"]
  line_colors : list
   colors of lines per directory e.g. ["red", "blue", "green"]
  line_styles : list
   styles of lines per directory e.g. ["-","--","-."]
  """
  _, axes = plt.subplots(2, 2)
  for i in range(0, 2):
    for j in range(0, 2):
      sphere_index = 2 * i + j
      for index in range(0, len(legend)):
        axes[i, j].plot(
            results[sphere_index, :, 0, index],
            results[sphere_index, :, 2, index],
            color=line_colors[index],
            linestyle=line_styles[index]
        )
      axes[i, j].set_title(spheres[sphere_index])
      axes[i, j].set_ylabel('CRC value')
      axes[i, j].set_xlabel('BV value')
      axes[i, j].legend((legend))
      axes[i, j].grid()
  plt.tight_layout()
  plt.show()


def sanity_checks(args, input_directories_paths):
  """
  Performs sanity checks:
  1. for each directory we require information about lines's description, color and style
  2. all given directories has to exist

  In case when is only one directory provided line's parameters are not required - if any of them is missing
  this method add default value (description is name of directory, color is blue, type is continuous line).

  Parameters
  ----------
  args : argparse.Namespace
   populated namespace
  input_directories_paths : list
   each entry is Path object which is a path to input directory with *.hdr files

  Returns
  ------
  all parameters are correct  : bool

  """

  #Handling single directory scenario (loot at above method description)
  if len(args.input_directory) == 1:
    if args.line_description == []:
      args.line_description.append(str(input_directories_paths[0].stem))
    if args.line_color == []:
      args.line_color.append("blue")
    if args.line_style == []:
      args.line_style.append("-")

  control = [
      len(args.input_directory),
      len(args.line_description),
      len(args.line_color),
      len(args.line_style)
  ]
  if control != [len(args.input_directory)] * 4:
    print(
        "Error - number of given directories and line descriptions, colors and styles has to be equal."
    )
    return False

  input_directories_check = [
      idir.exists() and idir.is_dir() for idir in input_directories_paths
  ]
  if input_directories_check != [True] * len(input_directories_check):
    print(
        "Error - at least one input directory does not exist:", [
            args.input_directory[i]
            for i, check in enumerate(input_directories_check)
            if not check
        ]
    )
    return False
  return True


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      '-idir',
      '--input-directory',
      action='append',
      required=True,
      help="Path to directory with data files"
  )
  ap.add_argument(
      '-lc',
      '--line-color',
      action='append',
      default=[],
      help="Color of plot line for data from the directory"
  )
  ap.add_argument(
      '-ls',
      '--line-style',
      action='append',
      default=[],
      help="Line type of plot for data from the directory"
  )
  ap.add_argument(
      '-ld',
      '--line-description',
      action='append',
      default=[],
      help="Title of plotted line for data from the directory"
  )
  ap.add_argument(
      "-from", "--from-iteration", type=int, help="From which iteration start"
  )
  ap.add_argument(
      "-to", "--to-iteration", type=int, help="To which iteration continue"
  )

  args = ap.parse_args()
  input_directories_paths = [Path(idir) for idir in args.input_directory]
  if not sanity_checks(args, input_directories_paths):
    sys.exit(1)

  results, spheres = generate_results(
      input_directories_paths, args.from_iteration, args.to_iteration
  )
  plot_results(
      results, spheres, args.line_description, args.line_color, args.line_style
  )


if __name__ == "__main__":
  main()
