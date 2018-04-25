import csv
import json
from pprint import pprint


VX0_MIN = -18.885
VX0_MAX = 18.065
VY0_MIN = -152.477
VY0_MAX = -86.374
VZ0_MIN = -15.646
VZ0_MAX = 9.974
AX_MIN = -48.0287647107959
AX_MAX = 30.302
AY_MIN = 9.723
AY_MAX = 49.18
AZ_MIN = -52.43
AZ_MAX = 2.95522851438373
PFX_X_MIN = -25.438405804093442
PFX_X_MAX = 17.2
PFX_Z_MIN = -15.24
PFX_Z_MAX = 18.84426818102172
START_SPEED_MIN = 59
START_SPEED_MAX = 104.4


def normalize(x, min, max):
  return (x - min) / (max - min)


# TODO - refactor this
def convert_json(file):
  with open(file) as f:
    content = f.readlines()

  fields = [
    {'key': 'vx0', 'min': VX0_MIN, 'max': VX0_MAX},
    {'key': 'vy0', 'min': VY0_MIN, 'max': VY0_MAX},
    {'key': 'vz0', 'min': VZ0_MIN, 'max': VZ0_MAX},
    {'key': 'ax', 'min': AX_MIN, 'max': AX_MAX},
    {'key': 'ay', 'min': AY_MIN, 'max': AY_MAX},
    {'key': 'az', 'min': AZ_MIN, 'max': AZ_MAX},
    {'key': 'start_speed', 'min': START_SPEED_MIN, 'max': START_SPEED_MAX},
    {'key': 'left_handed_pitcher', 'min': None, 'max': None},
    {'key': 'pitch_code', 'min': None, 'max': None}]

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

      writer.writerow(pitch_fields)


convert_json("pitch_type_training_data.json")
convert_json("pitch_type_test_data.json")
