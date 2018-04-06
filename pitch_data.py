import tensorflow as tf

csv_column_types = [
  [''], # des (0)
  [],   # id (1)
  [''], # type (2)
  [''], # code (3)
  [],   # tfs (4)
  [''], # tfs_zulu (5)
  [],   # x (6)
  [],   # y (7)
  [],   # event_num (8)
  [''], # sv_id (9)
  [''], # play_guid (10)
  [],   # start_speed (11)
  [],   # end_speed (12)
  [],   # sz_top (13)
  [],   # sz_bot (14)
  [],   # pfx_x (15)
  [],   # pfx_z (16)
  [],   # px (17)
  [],   # pz (18)
  [],   # x0 (19)
  [],   # y0 (20)
  [],   # z0 (21)
  [],   # vx0 (22)
  [],   # vy0 (23)
  [],   # vz0 (24)
  [],   # ax (25)
  [],   # ay (26)
  [],   # az (27)
  [],   # break_y (28)
  [],   # break_angle (29)
  [],   # break_length (30)
  [''], # pitch_type (31)
  [0],  # pitch_code (32)
  [],   # type_confidence (33)
  [],   # zone (34)
  [],   # nasty (35)
  [],   # spin_dir (36)
  [],   # spin_rate (37)
]


def decode_csv(line):
  parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)

  pitch_type = parsed_line[31]
  pitch_code = parsed_line[32]
  pitch_type_label = tf.one_hot(pitch_code, 12)

  break_y = parsed_line[28]
  break_angle = parsed_line[29]
  break_length = parsed_line[30]
  spin_rate = parsed_line[35]
  start_speed = parsed_line[11]
  end_speed = parsed_line[12]

  data = tf.stack([break_y, break_angle, break_length])
  return pitch_type, pitch_type_label, data


def load_data(filename):
  dataset = tf.data.TextLineDataset([filename])
  dataset = dataset.skip(1)
  dataset = dataset.map(decode_csv)
  dataset = dataset.batch(100)
  return dataset

