import tensorflow as tf

# Pitch CSV column definitions
csv_column_types = [
  [''], # des (0)
  [],   # id (1)
  [''], # type (2)
  [''], # tfs_zulu (3)
  [],   # x (4)
  [],   # y (5)
  [],   # start_speed (6)
  [],   # end_speed (7)
  [],   # sz_top (8)
  [],   # sz_bot (9)
  [],   # pfx_x (10)
  [],   # pfx_z (11)
  [],   # px (12)
  [],   # pz (13)
  [],   # x0 (14)
  [],   # y0 (15)
  [],   # z0 (16)
  [],   # vx0 (17)
  [],   # vy0 (18)
  [],   # vz0 (19)
  [],   # ax (20)
  [],   # ay (21)
  [],   # az (22)
  [],   # break_y (23)
  [],   # break_angle (24)
  [],   # break_length (25)
  [''], # pitch_type (26)
  [0],  # pitch_code (27)
  [],   # type_confidence (28)
  [],   # zone (29)
  [],   # nasty (30)
  [],   # spin_dir (31)
  [],   # spin_rate (32)
]


PITCH_CLASSES = [
  'Fastball (two-seam)',
  'Fastball (four-seam)',
  'Fastball (sinker)',
  'Fastball (cutter)',
  'Slider',
  'Changeup',
  'Curveball']

VX0_MIN = -20.854
VX0_MAX = 19.447
VY0_MIN = -152.477
VY0_MAX = -65.33
VZ0_MIN = -20.64
VZ0_MAX = 17.856
AX_MIN = -59.290089729458
AX_MAX = 40.181
AY_MIN = 8.32652554182805
AY_MAX = 49.18
AZ_MIN = -52.756
AZ_MAX = 8.645

def col_keys():
  return [
    'vx0',
    'vy0',
    'vz0',
    'ax',
    'ay',
    'az',
  ]


def estimator_cols():
  return [
    tf.feature_column.numeric_column(
      key='vx0',
      normalizer_fn=lambda x: ((x - VX0_MIN) / (VX0_MAX - VX0_MIN))),

    tf.feature_column.numeric_column(
      key='vy0',
      normalizer_fn=lambda x: ((x - VY0_MIN) / (VY0_MAX - VY0_MIN))),

    tf.feature_column.numeric_column(
      key='vz0',
      normalizer_fn=lambda x: ((x - VZ0_MIN) / (VZ0_MAX - VZ0_MIN))),

    tf.feature_column.numeric_column(
      key='ax',
      normalizer_fn=lambda x: ((x - AX_MIN) / (AX_MAX - AX_MIN))),

    tf.feature_column.numeric_column(
      key='ay',
      normalizer_fn=lambda x: ((x - AY_MIN) / (AY_MAX - AY_MIN))),

    tf.feature_column.numeric_column(
      key='az',
      normalizer_fn=lambda x: ((x - AZ_MIN) / (AZ_MAX - AZ_MIN))),
  ]


def decode_csv_est(line):
  parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)

  pitch_code = parsed_line[27]

  break_y = parsed_line[23]
  break_angle = parsed_line[24]
  break_length = parsed_line[25]

  ax = parsed_line[20]
  ay = parsed_line[21]
  az = parsed_line[22]

  vx0 = parsed_line[17]
  vy0 = parsed_line[18]
  vz0 = parsed_line[19]

  px = parsed_line[12]
  pz = parsed_line[13]

  start_speed = parsed_line[6]
  end_speed = parsed_line[7]

  conf = parsed_line[28]

  features = dict(zip(col_keys(), [
      vx0,
      vy0,
      vz0,
      ax,
      ay,
      az,
      ]))

  return features, pitch_code


def csv_input_fn(filename, batchsize=100):
  dataset = tf.data.TextLineDataset([filename]).skip(1)
  dataset = dataset.map(decode_csv_est)
  dataset = dataset.shuffle(10000).repeat().batch(batchsize)
  return dataset


def csv_eval_fn(filename, batchsize=100):
  dataset = tf.data.TextLineDataset([filename]).skip(1)
  dataset = dataset.map(decode_csv_est)
  dataset = dataset.shuffle(1000).batch(batchsize)
  return dataset


def test_pitch():
  samples = [
    [3.183,-135.149,-2.005,-16.296,29.576,-22.414],
    [-9.706,-135.569,-4.866,6.548,28.308,-14.883],
    [-3.94,-132.87,-1.45,18.93,30.41,-31.62],
    [7.254,-133.116,-6.822,2.01,30.686,-8.24],
    [-9.375,-115.324,-2.395,-1.281,22.358,-39.066],
    [5.09,-123.243,-6.224,-19.159,28.283,-25.269],
    [0.664,-117.548,1.539,3.957,24.355,-40.877],
  ]

  features = dict(zip(col_keys(), samples))
  return tf.data.Dataset.from_tensors(features)
