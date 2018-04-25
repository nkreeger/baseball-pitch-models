import csv
import json
from pprint import pprint

PX_MIN = -5.08314439875463
PX_MAX = 4.252838549
PZ_MIN = -3.14564239312703
PZ_MAX = 6.06644249133382
SZ_TOP_MIN = 2.6534936194861984
SZ_TOP_MAX = 4.43416657143635
SZ_BOT_MIN = 1.068002693284189
SZ_BOT_MAX = 2.434942748330954


def normalize(x, min, max):
  return (x - min) / (max - min)


# TODO - refactor this
def convert_json(file):
  with open(file) as f:
    content = f.readlines()

  fields = [
    {'key': 'px', 'min': PX_MIN, 'max': PX_MAX},
    {'key': 'pz', 'min': PZ_MIN, 'max': PZ_MAX},
    {'key': 'sz_top', 'min': SZ_TOP_MIN, 'max': SZ_TOP_MAX},
    {'key': 'sz_bot', 'min': SZ_BOT_MIN, 'max': SZ_BOT_MAX},
    {'key': 'left_handed_batter', 'min': None, 'max': None}]

  with open(file + '.csv', 'w') as f:
    writer = csv.writer(f)
    for chunk in enumerate(content):
      pitch = json.loads(chunk[1])
      pitch_fields = []
      for field in enumerate(fields):
        ft = field[1]
        if (ft['min'] != None and ft['max'] != None):
          pitch_fields.append(normalize(pitch[ft['key']], ft['min'], ft['max']))
        else:
          pitch_fields.append(pitch[ft['key']])

      # Special label field for SZ:
      if (pitch['type'] == "S"):
        pitch_fields.append(0)
      else:
        pitch_fields.append(1)
      writer.writerow(pitch_fields)


convert_json("strike_zone_training_data.json")
convert_json("strike_zone_test_data.json")