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


VX0_MIN = -23.161
VX0_MAX = 19.498
VY0_MIN = -150.782746752869
VY0_MAX = -62.356
VZ0_MIN = -20.95
VZ0_MAX = 27.815
AX_MIN = -38.273
AX_MAX = 33.646
AY_MIN = 3.46
AY_MAX = 53.299
AZ_MIN = -61.86
AZ_MAX = 23.78
PX_MIN = -8.764
PX_MAX = 12.9529095060724
PZ_MIN = -4.79923819378
PZ_MAX = 12.488540954706
X0_MIN = -5.87467296544683
X0_MAX = 9.717
Z0_MIN = -0.0160436028988832
Z0_MAX = 9.7475265071709

def col_keys():
  return [
    'vx0',
    'vy0',
    'vz0',
    'ax',
    'ay',
    'az',
    'px',
    'pz',
    'x0',
    'z0'
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

    tf.feature_column.numeric_column(
      key='px',
      normalizer_fn=lambda x: ((x - PX_MIN) / (PX_MAX - PX_MIN))),

    tf.feature_column.numeric_column(
      key='pz',
      normalizer_fn=lambda x: ((x - PZ_MIN) / (PZ_MAX - PZ_MIN))),

    tf.feature_column.numeric_column(
      key='x0',
      normalizer_fn=lambda x: ((x - X0_MIN) / (X0_MAX - X0_MIN))),

    tf.feature_column.numeric_column(
      key='z0',
      normalizer_fn=lambda x: ((x - Z0_MIN) / (Z0_MAX - Z0_MIN))),
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

  x0 = parsed_line[14]
  z0 = parsed_line[16]

  conf = parsed_line[28]

  features = dict(zip(col_keys(), [
      vx0,
      vy0,
      vz0,
      ax,
      ay,
      az,
      px,
      pz,
      x0,
      z0
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
    [2.001,-134.272,-1.451,-19.153,30.434,-24.366,-1.93,3.192,-1.318,5.481],
    [-6.409,-136.065,-3.995,7.665,34.685,-11.96,0.416,2.963,2.28,5.302],
    [-8.704,-132.38,-2.685,17.411,26.452,-22.438,0.2,2.643,2.253,5.301],
    [4.243,-127.708,-6.167,2.58,24.596,-22.981,0.687,1.97,-1.193,6.206],
    [1.331,-123.126,-5.872,3.406,22.882,-29.909,-0.133,1.1,-0.966,6.025],
    [-8.545,-121.495,-3.271,16.668,25.539,-25.448,-0.028,1.778,2.088,5.372],
    [-6.309,-110.409,0.325,-10.28,21.774,-34.111,-1.821,2.083,2.179,5.557],
  ]

  features = dict(zip(col_keys(), samples))
  return tf.data.Dataset.from_tensors(features)
