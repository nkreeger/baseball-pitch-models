import tensorflow as tf

import pitch_data


def custom_classifier(features, labels, mode, params):
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # predictions:
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def model(model_dir):
  return tf.estimator.DNNClassifier(
          feature_columns=pitch_data.estimator_cols(),
          hidden_units=[150, 125, 100, 75, 50, 25],
          activation_fn=tf.nn.relu,
          n_classes=7,
          optimizer=tf.train.AdamOptimizer(),
          dropout=0.1,
          model_dir='models')

  # return tf.estimator.Estimator(
  #   model_fn=custom_classifier,
  #   params={
  #     'feature_columns': cols,
  #     'hidden_units': [250, 125, 75, 25],
  #     'n_classes': 10
  #   })
