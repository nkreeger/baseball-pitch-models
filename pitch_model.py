import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
import tensorflow.contrib.eager as tfe  # pylint: disable=g-bad-import-order

csv_column_types = [
  [''], # des
  [],   # id
  [''], # type
  [''], # code
  [],   # tfs
  [''], # tfs_zulu
  [],   # x
  [],   # y
  [],   # event_num
  [''], # sv_id
  [''], # play_guid
  [],   # start_speed
  [],   # start_speed
  [],   # sz_top
  [],   # sz_bot
  [],   # pfx_x
  [],   # pfx_z
  [],   # px
  [],   # pz
  [],   # x0
  [],   # y0
  [],   # z0
  [],   # vx0
  [],   # vy0
  [],   # vz0
  [],   # ax
  [],   # ay
  [],   # az
  [],   # break_y
  [],   # break_angle
  [],   # break_length
  [''], # pitch_type
  [],   # type_confidence
  [],   # zone
  [],   # nasty
  [],   # spin_dir
  [],   # spin_rate
]

def decode_csv(line):
  parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)
  # TODO(kreeger): Exclude non-needed fields.
  return parsed_line

def load_training_data():
  dataset = tf.data.TextLineDataset(['training_data.csv'])
  dataset = dataset.skip(1)
  dataset = dataset.map(decode_csv)
  dataset = dataset.batch(25)

  iterator = tfe.Iterator(dataset)
  print('')
  print iterator.next()

def main(argv):
  tfe.enable_eager_execution()

  load_training_data()

if __name__ == '__main__':
  main(argv=sys.argv)
