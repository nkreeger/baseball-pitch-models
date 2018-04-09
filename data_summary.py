import sys
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from pitch_data import load_data,NUM_PITCH_CLASSES


def main(argv):
  tfe.enable_eager_execution()

  dataset = load_data('2015_pitches.csv', 1)

  # TODO(kreeger): This can be vectorized.
  buckets = []
  for _ in range(NUM_PITCH_CLASSES):
    buckets.append({
        'count':0,
        'd': tf.zeros([1,10], tf.float32)
        })

  entries = 0
  for (batch, (label, data)) in enumerate(tfe.Iterator(dataset)):
    pitch_type = tf.argmax(label, axis=1, output_type=tf.int64)
    idx = int(pitch_type)
    entries = entries +1
    buckets[idx]['count'] = buckets[idx]['count'] + 1
    buckets[idx]['d'] = tf.add(buckets[idx]['d'], data)

  print ('entries: {}'.format(entries))
  print ('')
  index = 0
  for bucket in buckets:
    count = bucket['count']
    output = '{},{},'.format(index, count)
    for idx in xrange(int(bucket['d'].shape[1])):
      item = float(bucket['d'][0][idx])
      if item == 0.0:
        output = output + '0,'
      else:
        output = output + '{},'.format(item / count)

    print (output)
    index = index + 1
    # print('{},{},{},{},{}'.format(break_y, break_angle, break_length, pfx_x, spin_rate))
  


if __name__ == '__main__':
  main(argv=sys.argv)

