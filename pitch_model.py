import math
import sys
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import pitch_data


class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

    initializer = tf.initializers.random_uniform(0.1, 11)

    self.dense1 = tf.layers.Dense(50, activation=tf.nn.relu)
    self.dense2 = tf.layers.Dense(25, activation=tf.nn.relu)
    self.dense3 = tf.layers.Dense(11, activation=tf.nn.relu)
    self.dropout = tf.layers.Dropout(0.5)

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
  for (batch, (labels, data)) in enumerate(tfe.Iterator(dataset)):
    train_loop(model, optimizer, step_counter, batch, labels, data)


def train_same_batch(model, optimizer, labels, data, step_counter):
  for idx in range(100):
    train_loop(model, optimizer, step_counter, idx, labels, data)


def train_loop(model, optimizer, step_counter, batch, labels, data):
  with tfe.GradientTape() as tape:
    logits = model(data, training=True)
    loss_value = loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)

  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)

  if batch % 100 == 0:
    print(' - Step #%d\tLoss: %.6f' % (batch, loss_value))


def test(model, labels, data):
  logits = model(data, training=False)
  accuracy = compute_accuracy(logits, labels)
  print(' --> Test accuracy: %.2f' % (accuracy))


def run_eager(argv):
  tfe.enable_eager_execution()

  model = Model()
  train_dataset = pitch_data.load_data('training_data.csv', 50)
  test_dataset = pitch_data.load_data('test_data.csv', 50)
  test_labels, test_data = tfe.Iterator(test_dataset).next()

  step_counter = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  # optimizer = tf.train.MomentumOptimizer(0.01, 0.5)

  for _ in range(100):
    start = time.time()
    # TODO(kreeger): Gate this.
    # train_same_batch(model, optimizer, test_labels, test_data, step_counter)
    train(model, optimizer, train_dataset, step_counter)
    end = time.time()
    print(' ** Train time for epoch #%d (%d total steps): %f' % (_ + 1, step_counter.numpy(), end - start))
    test(model, test_labels, test_data)
    print('')


def main(argv):
  col_names = pitch_data.estimator_cols()
  cols = []
  for name in col_names:
    cols.append(tf.feature_column.numeric_column(key=name))

  classifier = tf.estimator.DNNClassifier(
          feature_columns=cols,
          hidden_units=[100, 75, 50, 25],
          n_classes=11,
          model_dir='models')

  for _ in range(100):
    print('------ TRAIN ----------: {}'.format(_))
    classifier.train(
            input_fn=lambda:pitch_data.csv_input_fn('training_data.csv', batchsize=100), steps=1000)

    print('------ EVALUATE ----------: {}'.format(_))
    eval_result = classifier.evaluate(
            input_fn=lambda:pitch_data.csv_eval_fn('test_data.csv'))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

  # FF / 0.924
  predictions = classifier.predict(input_fn=pitch_data.test_pitch)
  template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

  print('\n')
  expected = ['FF']
  for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(class_id, 100 * probability, expec))

  print('\n')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # run_eager(argv=sys.argv)
  tf.app.run(main)
