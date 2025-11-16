"""
TODO: add docstring
"""
import json
import copy
from pprint import pprint
import os


def get_translation(_dict, volume):
  """Change units cm -> mm, rotate image (imshow)"""
  return -10 * _dict[volume]['setTranslation'][0], 10 * _dict[volume]['setTranslation'][1], 10 * \
         _dict[volume]['setTranslation'][2]


class PhantomParser():
  """
    Class parsing GATE phantom to python dict
    """

  def __init__(self, file_name_tree, file_name_phantom, file_name_type):

    with open(file_name_tree, 'r', encoding="utf-8") as outfile:
      self.tree = json.load(outfile)
    self.file_name_phantom = file_name_phantom
    self.file_name_type = file_name_type
    self._mapa = {}
    self.geometry = {}
    self._join_dict()

  def _path(self, _dict, parent=None):
    for key, value in _dict.items():
      self._mapa[key] = parent

      if isinstance(value, dict):
        self._path(value, key)
      else:
        pass

  def _path_volume(self, volume):
    _list = [volume]
    w = self._mapa.get(volume)
    while w is not None:
      _list.append(w)
      w = self._mapa.get(w)
    _list.reverse()
    return _list

  def _geo_parser(self):
    temp_dict = {}
    with open(self.file_name_type, 'r', encoding="utf-8") as outfile:
      for line in outfile.readlines():
        a = line.split('\t')
        temp_dict[a[0]] = {"material": a[1], "type": a[2].rstrip()}

    return temp_dict

  def _mac_parser(self):
    temp_dict = {}
    with open(self.file_name_phantom, 'r', encoding="utf-8") as outfile:
      for line in outfile.readlines():
        if 'geometry' in line:
          a = line.split('/')
          b = a[-1].split()
          if temp_dict.get(a[2]):
            temp_dict[a[2]].update({b[0]: float(b[1]), b[0] + '_units': b[-1]})
          else:
            temp_dict[a[2]] = {}
            temp_dict[a[2]].update({b[0]: float(b[1]), b[0] + '_units': b[-1]})

        if 'placement' in line:
          a = line.split('/')
          b = a[-1].split()
          if temp_dict.get(a[2]):
            temp_dict[a[2]].update(
                {
                    b[0]: (float(b[1]), float(b[2]), float(b[3])),
                    b[0] + '_units': b[-1]
                }
            )
          else:
            temp_dict[a[2]] = {}
            temp_dict[a[2]].update(
                {
                    b[0]: (float(b[1]), float(b[2]), float(b[3])),
                    b[0] + '_units': b[-1]
                }
            )

    return temp_dict

  def _join_dict(self):
    self._path(self.tree)

    self.geometry = self._mac_parser()
    geometry2 = self._geo_parser()

    def sum_v(v_1, v_2):
      return (v_1[0] + v_2[0], v_1[1] + v_2[1], v_1[2] + v_2[2])

    g_temp = copy.deepcopy(self.geometry)
    g_temp['world'] = {}
    g_temp['world']['setTranslation'] = (0, 0, 0)

    for key, value in self.geometry.items():
      for volume in self._path_volume(key)[:-1]:
        value['setTranslation'] = sum_v(
            value['setTranslation'], g_temp[volume]['setTranslation']
        )
      value['factor'] = len(self._path_volume(key)[:-1])

    for key, value in self.geometry.items():
      value.update(geometry2[key])


abs_path = os.path.abspath(os.path.dirname(__file__))
JPET = PhantomParser(
    os.path.join(abs_path, 'tree.txt'),
    os.path.join(abs_path, 'NEMA_IEC_2001_IQ_Phantom.mac'),
    os.path.join(abs_path, 'phantom.txt')
)

if __name__ == "__main__":
  JPET = PhantomParser(
      'tree.txt', 'NEMA_IEC_2001_IQ_Phantom.mac', 'phantom.txt'
  )
  pprint(JPET.geometry)
