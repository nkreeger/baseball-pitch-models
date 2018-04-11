import tensorflow as tf

import pitch_data


def model(model_dir):
  col_names = pitch_data.estimator_cols()
  cols = []
  for name in col_names:
    print('adding : {}'.format(name))
    cols.append(tf.feature_column.numeric_column(key=name))

  return tf.estimator.DNNClassifier(
          feature_columns=cols,
          hidden_units=[250, 125, 75, 25],
          n_classes=10,
          optimizer=tf.train.AdamOptimizer(),
          dropout=0.1,
          model_dir='models')
