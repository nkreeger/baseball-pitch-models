import sys
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from pitch_data import load_data


def main(argv):
  tfe.enable_eager_execution()

  dataset = load_data('test_data.csv', 1)

  # TODO(kreeger): This can be vectorized.
  buckets = []
  for _ in range(12):
    buckets.append(tf.zeros([3], tf.float32))

  entries = 0
  for (batch, (pitch_str, labels, data)) in enumerate(tfe.Iterator(dataset)):
    pitch_type = tf.argmax(labels, axis=1, output_type=tf.int64)
    pitch_type_idx = int(pitch_type)
    buckets[pitch_type_idx] = tf.add(buckets[pitch_type_idx], data)
    entries = entries + 1

  print ('entries: {}'.format(entries))
  print ('')
  index = 0
  for bucket in buckets:
    break_y = bucket[0][0]
    break_angle = bucket[0][1]
    break_length = bucket[0][2]
    print('{},{},{},{}'.format(index, break_y, break_angle, break_length))
    index = index + 1


if __name__ == '__main__':
  main(argv=sys.argv)

