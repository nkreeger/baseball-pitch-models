import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
import tensorflow.contrib.eager as tfe  # pylint: disable=g-bad-import-order

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
  [],   # type_confidence (32)
  [],   # zone (33)
  [],   # nasty (34)
  [],   # spin_dir (35)
  [],   # spin_rate (36)
]

# Pitch type definitions
# FA - 'Fastball' (0)
# FF - 'Four-seam Fastball' (1)
# FT - 'Two-seam Fastball' (2)
# FC - 'Fastball (cutter)' (3) 
# FS - 'Fastball (sinker)' (4)
# SI - 'Fastball (sinker)' (5)
# SF - 'Fastball (split-fingered)' (6)
# SL - 'Slider' (7)
# CB - 'Curveball' (8)
# CU - 'Curveball' (9)
# KC - 'Kunckle-curve' (10)
# KN - 'Knuckleball' (11)
# EP - 'Eephus' (12)
# PO - 'Pitch out' (13)
# FO - 'Pitch out' (14)
# UN - 'Unidentifed' (15)
# XX - 'Unidentifed' (16)

def decode_csv(line):
  parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)
  # Return pitch_type, label, vx0, vyz0, vz0, start_speed, end_speed
  pitch_type = parsed_line[31]
  pitch_type_label = tf.string_to_hash_bucket(pitch_type, 17)
  vx0 = parsed_line[22]
  vy0 = parsed_line[23]
  vz0 = parsed_line[24]
  start_speed = parsed_line[11]
  end_speed = parsed_line[12]
  return pitch_type, pitch_type_label, vx0, vy0, vz0, start_speed, end_speed

def load_training_data():
  dataset = tf.data.TextLineDataset(['training_data.csv'])
  dataset = dataset.skip(1)
  dataset = dataset.map(decode_csv)
  dataset = dataset.batch(5)
  return tfe.Iterator(dataset)

def model():
  #
  # TODO - left off right here. Need a combo of this and the mnist_eager
  #
  inputs = tf.keras.layers.Input((4,))
  dense1 = tf.keras.layers.Dense(10, use_bias=True, name='Dense1', activation='sigmoid')(inputs)
  dense2 = tf.keras.layers.Dense(17, use_bias=True, name='Dense2', activation='softmax')(dense1)
  model = tf.keras.models.Model(inputs=[inputs], outputs=[dense2])
  model.compile(loss='categorical_crossentropy', optimizer='adam')

def main(argv):
  tfe.enable_eager_execution()

  iterator = load_training_data()
  batch = iterator.next()
  # batch = iterator.next()
  # batch = iterator.next()
  print batch

  # 17 pitch types:
  # pitch_types = tf.one_hot(tf.string_to_number(batch[31]), 17) 
  # vx0 = batch[22]
  # vy0 = batch[23]
  # vz0 = batch[24]

  # print batch[31]
  # print pitch_types

  model_ = model()

  # with tfe.GradientTape() as tape:
  #   logits = model(iterator.next(), training=True) 
  #   print ('logits: {}'.format(logits))

if __name__ == '__main__':
  main(argv=sys.argv)
