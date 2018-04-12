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
  [],   # is lefty (33)
]


PITCH_CLASSES = [
  'Fastball (two-seam)',
  'Fastball (four-seam)',
  'Fastball (sinker)',
  'Fastball (cutter)',
  'Slider',
  'Changeup',
  'Curveball']


VX0_MIN = -22.026
VX0_MAX = 19.4878168193
VY0_MIN = -150.782746752869
VY0_MAX = -64.4392183150571
VZ0_MIN = -19.7706017189303
VZ0_MAX = 27.815
AX_MIN = -36.0593384891835
AX_MAX = 32.5001540833473
AY_MIN = 5.45147566871083
AY_MAX = 48.53
AZ_MIN = -61.86
AZ_MAX = 11.046748574341
PFX_X_MIN = -21.183690868172043
PFX_X_MAX = 19.336301479608238
PFX_Z_MIN = -70.206
PFX_Z_MAX = 33.16047741119556
START_SPEED_MIN = 45.1
START_SPEED_MAX = 103.5


def col_keys():
  return [
    'vx0',
    'vy0',
    'vz0',
    'ax',
    'ay',
    'az',
    'pfx_x',
    'pfx_z',
    'start_speed',
    'is_lefty'
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
      key='pfx_x',
      normalizer_fn=lambda x: ((x - PFX_X_MIN) / (PFX_X_MAX - PFX_X_MIN))),

    tf.feature_column.numeric_column(
      key='pfx_z',
      normalizer_fn=lambda x: ((x - PFX_Z_MIN) / (PFX_Z_MAX - PFX_Z_MIN))),

    tf.feature_column.numeric_column(
      key='start_speed',
      normalizer_fn=lambda x: ((x - START_SPEED_MIN) / (START_SPEED_MAX - START_SPEED_MIN))),

    tf.feature_column.numeric_column(key='is_lefty')
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

  pfx_x = parsed_line[10]
  pfx_z = parsed_line[11]
  is_left = parsed_line[33]

  conf = parsed_line[28]

  features = dict(zip(col_keys(), [
    vx0,
    vy0,
    vz0,
    ax,
    ay,
    az,
    pfx_x,
    pfx_z,
    start_speed,
    is_left
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
    [5.288672611,-139.2178033,-4.486851726,-21.23317238,30.97437438,-20.78893317,-10.80863952,5.795535677,95.9,0],
    [5.773371605,-137.7297696,0.007950181,-10.84800973,29.45790782,-18.78468501,-5.625247466,6.9430693009999995,94.9,0],
    [-7.407883119,-129.9573035,-3.935528779,16.05806237,29.26242568,-23.03361428,9.465174699,5.387686588,89.7,1],
    [-8.300764072,-127.790885,-4.138864138,-0.919059623,27.65359163,-18.73567624,-0.558779119,8.170396846,88.2,1],
    [5.831738959,-125.4155323,-4.837203731,12.91704723,27.57701826,-24.85489435,8.186487562,4.638688996,86.6,0],
    [4.646421639,-130.4076624,-2.558437493,-19.68619755,29.51564233,-27.21783694,-11.52598704,2.901790998,89.9,0],
    [-7.315775688,-106.030691,-2.015545536,-6.597280527,19.81207628,-33.8590337,-5.853302672,-1.494968722,73.1,1],
  ]

  features = dict(zip(col_keys(), samples))
  return tf.data.Dataset.from_tensors(features)
