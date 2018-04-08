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

NUM_PITCH_CLASSES = 11


def decode_csv(line):
  parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)

  pitch_code = parsed_line[27]
  label = tf.one_hot(pitch_code, NUM_PITCH_CLASSES)

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

  conf = parsed_line[28]

  data = tf.stack([ax, ay, az, vx0, vy0, vz0, break_y, break_angle, break_length, conf])
  return label, data


def load_data(filename, batchsize=100):
  dataset = tf.data.TextLineDataset([filename])
  dataset = dataset.skip(1)
  dataset = dataset.map(decode_csv)
  dataset = dataset.batch(batchsize)
  return dataset


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

  conf = parsed_line[28]

  cols = [
      # 'vx0',
      # 'vy0',
      # 'vz0',
      # 'ax',
      # 'ay',
      # 'az',
      'break_y',
      'break_angle',
      'break_length'
  ]
  features = dict(zip(cols, [
      # vx0,
      # vy0,
      # vz0,
      # ax,
      # ay,
      # az,
      break_y,
      break_angle,
      break_length
      ]))

  return features, pitch_code


def csv_input_fn(filename, batchsize=100):
  dataset = tf.data.TextLineDataset([filename]).skip(1)
  dataset = dataset.map(decode_csv_est)
  dataset = dataset.shuffle(1000).repeat().batch(batchsize)
  return dataset


def csv_eval_fn(filename, batchsize=100):
  dataset = tf.data.TextLineDataset([filename]).skip(1)
  dataset = dataset.map(decode_csv_est)
  dataset = dataset.batch(batchsize)
  return dataset

