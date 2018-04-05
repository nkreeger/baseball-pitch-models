import sys
import time

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

  data = tf.stack([break_y, break_angle, break_length, spin_rate])
  return pitch_type, pitch_type_label, data


def load_training_data():
  dataset = tf.data.TextLineDataset(['training_data.csv'])
  dataset = dataset.skip(1)
  dataset = dataset.map(decode_csv)
  dataset = dataset.batch(10)
  return dataset


class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

    self.dense1 = tf.layers.Dense(12, use_bias=True, name='dense1', activation=tf.nn.relu)
    self.dense2 = tf.layers.Dense(24, use_bias=True, name='dense2', activation=tf.nn.relu)
    self.dense3 = tf.layers.Dense(12, use_bias=True, name='dense3', activation=tf.nn.relu)

  def __call__(self, inputs, training):
    y = self.dense1(inputs)
    y = self.dense2(y)
    y = self.dense3(y)
    return y


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
  labels = tf.argmax(labels, axis=1, output_type=tf.int64)
  batch_size = int(logits.shape[0])
  return tf.reduce_sum(
      tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def loss(logits, labels):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))


def train(model, optimizer, dataset, step_counter):
  start = time.time()

  # pitch_str, labels, data = tfe.Iterator(dataset).next()
  # for batch in range(5000):
  for (batch, (pitch_str, labels, data)) in enumerate(tfe.Iterator(dataset)):
    with tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=step_counter):
      with tfe.GradientTape() as tape:
        logits = model(data, training=True)
        loss_value = loss(logits, labels)
        accuracy = compute_accuracy(logits, labels)

        tf.contrib.summary.scalar('loss', loss_value)
        tf.contrib.summary.scalar('accuracy', accuracy)

      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)

      if batch % 100 == 0:
        accuracy = compute_accuracy(logits, labels)
        rate = 100 / (time.time() - start)
        print(' - Step #%d\tLoss: %.6f, Accur: %.2f (%d steps/sec)' % (batch, loss_value, accuracy, rate))
        start = time.time()

  # print(pitch_str)
  # print(tf.argmax(labels, axis=1, output_type=tf.int64))
  # test = model(data, training=False)
  # print(tf.argmax(test, axis=1, output_type=tf.int64))


def debug_dataset(dataset):
  for (batch, (pitch_str, labels, data)) in enumerate(tfe.Iterator(dataset)):
    print('%d - ' % batch)


def main(argv):
  tfe.enable_eager_execution()

  model = Model()
  dataset = load_training_data()
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.5)

  step_counter = tf.train.get_or_create_global_step()
  summary_writer = tf.contrib.summary.create_file_writer(None, flush_millis=10000)

  # with summary_writer.as_default():
  #   train(model, optimizer, dataset, step_counter)

  for _ in range(100):
    start = time.time()
    with summary_writer.as_default():
      train(model, optimizer, dataset, step_counter)
    end = time.time()
    print(' ** Train time for epoch #%d (%d total steps): %f\n' % (_ + 1, step_counter.numpy(), end - start))


if __name__ == '__main__':
  main(argv=sys.argv)
