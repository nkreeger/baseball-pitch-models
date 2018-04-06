import math
import sys
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from pitch_data import load_data,NUM_DATA_INPUTS,NUM_PITCH_CLASSES

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

    self.dense1 = tf.layers.Dense(44, activation=tf.nn.relu)
    self.dense2 = tf.layers.Dense(22, activation=tf.nn.relu)
    self.dense3 = tf.layers.Dense(11, activation=tf.nn.relu)
    # self.dropout = tf.layers.Dropout(0.5)

  def __call__(self, inputs, training):
    y = self.dense1(inputs)
    y = self.dense2(y)
    y = self.dense3(y)
    # y = self.dropout(y, training=training)
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
  for (batch, (pitch_str, labels, data)) in enumerate(tfe.Iterator(dataset)):
    train_loop(model, optimizer, step_counter, batch, labels, data)


def train_same_batch(model, optimizer, pitch_str, labels, data, step_counter):
  start = time.time()
  for idx in range(100):
    train_loop(model, optimizer, step_counter, idx, labels, data)


def train_loop(model, optimizer, step_counter, batch, labels, data):
  with tfe.GradientTape() as tape:
    logits = model(data, training=True)
    loss_value = loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)

  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)

  if batch % 500 == 0:
    accuracy = compute_accuracy(logits, labels)
    print(' - Step #%d\tLoss: %.6f, Accur: %.2f' % (batch, loss_value, accuracy))
    start = time.time()


def test(model, labels, data):
  logits = model(data, training=False)
  accuracy = compute_accuracy(logits, labels)
  print(' --> Test accuracy: %.2f' % (accuracy))


def main(argv):
  tfe.enable_eager_execution()

  model = Model()
  train_dataset = load_data('training_data.csv', 100)
  test_dataset = load_data('test_data.csv', 100)
  test_pitch_str, test_labels, test_data = tfe.Iterator(test_dataset).next()

  step_counter = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

  for _ in range(100):
    start = time.time()
    # TODO(kreeger): Gate this.
    # train_same_batch(model, optimizer, test_pitch_str, test_labels, test_data, step_counter)
    train(model, optimizer, train_dataset, step_counter)
    end = time.time()
    print(' ** Train time for epoch #%d (%d total steps): %f' % (_ + 1, step_counter.numpy(), end - start))
    test(model, test_labels, test_data)
    print('')


if __name__ == '__main__':
  main(argv=sys.argv)
