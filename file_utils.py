# Ported from PA4

import os
import json


def read_file(path):
  if os.path.isfile(path):
    with open(path) as json_file:
      data = json.load(json_file)
    return data
  else:
    raise Exception("file doesn't exist: ", path)


def read_file_in_dir(root_dir, file_name):
  path = os.path.join(root_dir, file_name)
  return read_file(path)
