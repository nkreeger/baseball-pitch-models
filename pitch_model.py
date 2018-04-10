import math
import sys
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import pitch_data


def main(argv):
  col_names = pitch_data.estimator_cols()
  cols = []
  for name in col_names:
    cols.append(tf.feature_column.numeric_column(key=name))

  classifier = tf.estimator.DNNClassifier(
          feature_columns=cols,
          hidden_units=[300, 200, 100, 75, 50, 25],
          n_classes=11,
          optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001),
          model_dir='models')

  for _ in range(1000):
    print('------ TRAIN ----------: {}'.format(_))
    classifier.train(
            input_fn=lambda:pitch_data.csv_input_fn('training_data.csv', batchsize=100),
            steps=1000)

    print('------ EVALUATE ----------: {}'.format(_))
    eval_result = classifier.evaluate(
            input_fn=lambda:pitch_data.csv_eval_fn('test_data.csv'))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    if _ % 10 == 0:
      # FF / 0.924
      predictions = classifier.predict(input_fn=pitch_data.test_pitch)
      template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

      print('\n')
      expected = ['FF', 'SL', 'FT', 'CH', 'KN', 'CU', 'EP', 'FS', 'KC', 'SI', 'FC']
      for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(pitch_data.PITCH_CLASSES[class_id], 100 * probability, expec))

      print('\n')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # run_eager(argv=sys.argv)
  tf.app.run(main)
