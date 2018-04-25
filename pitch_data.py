import tensorflow as tf
import numpy as np


# Pitch CSV column definitions
csv_column_types = [
  [''], # des (0)
  [],   # id (1)
  [''], # type (2)
  [''], # code (3)
  [''], # tfs_zulu (4)
  [],   # x (5)
  [],   # y (6)
  [],   # start_speed (7)
  [],   # end_speed (8)
  [],   # sz_top (9)
  [],   # sz_bot (10)
  [],   # pfx_x (11)
  [],   # pfx_z (12)
  [],   # px (13)
  [],   # pz (14)
  [],   # x0 (15)
  [],   # y0 (16)
  [],   # z0 (17)
  [],   # vx0 (18)
  [],   # vy0 (19)
  [],   # vz0 (20)
  [],   # ax (21)
  [],   # ay (22)
  [],   # az (23)
  [],   # break_y (24)
  [],   # break_angle (25)
  [],   # break_length (26)
  [''], # pitch_type (27)
  [0],  # pitch_code (28)
  [],   # type_confidence (29)
  [],   # zone (30)
  [],   # nasty (31)
  [],   # spin_dir (32)
  [],   # spin_rate (33)
  [],   # is lefty (34)
]


PITCH_CLASSES = [
  'Fastball (two-seam)',
  'Fastball (four-seam)',
  'Fastball (sinker)',
  'Fastball (cutter)',
  'Slider',
  'Changeup',
  'Curveball']


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


def load_np_data(filename):
  with open(filename) as f:
    content = f.readlines()

  features = []
  labels = []
  for line in content:
    items = line.split(',')

    f = [float(x) for x in items[:7]]
    features.append(f)

    labels.append(int(items[7]))
  
  combo = list(zip(features, labels))
  np.random.shuffle(combo)

  features, labels = zip(*combo)
  return (np.array(features, dtype=np.float32), _to_one_hot(labels, 7))


def _to_one_hot(indices, num_classes):
  one_hot = np.zeros([len(indices), num_classes], dtype=np.float32)
  one_hot[np.arange(len(indices)), indices] = 1
  return one_hot


def col_keys():
  return [
    'vx0',
    'vy0',
    'vz0',
    'ax',
    'ay',
    'az',
    # 'pfx_x',
    # 'pfx_z',
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

    # tf.feature_column.numeric_column(
    #   key='pfx_x',
    #   normalizer_fn=lambda x: ((x - PFX_X_MIN) / (PFX_X_MAX - PFX_X_MIN))),

    # tf.feature_column.numeric_column(
    #   key='pfx_z',
    #   normalizer_fn=lambda x: ((x - PFX_Z_MIN) / (PFX_Z_MAX - PFX_Z_MIN))),

    tf.feature_column.numeric_column(
      key='start_speed',
      normalizer_fn=lambda x: ((x - START_SPEED_MIN) / (START_SPEED_MAX - START_SPEED_MIN))),

    tf.feature_column.numeric_column(key='is_lefty')
  ]


def decode_csv_est(line):
  parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)

  pitch_code = parsed_line[28]

  vx0 = parsed_line[18]
  vy0 = parsed_line[19]
  vz0 = parsed_line[20]

  ax = parsed_line[21]
  ay = parsed_line[22]
  az = parsed_line[23]

  px = parsed_line[13]
  pz = parsed_line[14]

  start_speed = parsed_line[7]

  is_left = parsed_line[34]

  features = dict(zip(col_keys(), [
    vx0,
    vy0,
    vz0,
    ax,
    ay,
    az,
    # pfx_x,
    # pfx_z,
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


def test_pitch(batch_size):
  vx0 = [6.68052308575228,4.5090120321,12.1162969448299,6.68704669985913,-7.33881377525805,6.33455859698314,-0.503830311270123]
  vy0 = [-134.215511616881,-137.737724216,-134.319017263456,-127.784081756208,-120.756423391954,-123.728995275056,-113.078824317473]
  vz0 = [-6.35565979491619,-10.3847655281,-1.67805093859504,-2.16605968711717,-4.17767404309023,-9.40943912316956,0.0968908433095687]
  ax = [-19.6602769147989,-10.198026218,-23.8527109110688,-4.8155191192663,-0.534584499580229,-14.9072045358159,0.087745286968979]
  ay = [26.7031848314466,31.4999001822,27.8548664953099,25.3073119760175,22.29016452328,22.5933169332035,19.6934040447792]
  az = [-14.3430602022656,-10.5561059796,-21.8874374073744,-18.5761955457914,-29.146345314046,-18.3240910891258,-30.9850138322737]
  start_speed = [92.4,95.1,92.8,88.1,83.3,85.3,77.8]
  is_left = [0,0,0,0,1,0,0]

  samples = {
    'vx0': vx0,
    'vy0': vy0,
    'vz0': vz0,
    'ax': ax,
    'ay': ay,
    'az': az,
    'start_speed': start_speed,
    'is_lefty': is_left 
  }

  return tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
