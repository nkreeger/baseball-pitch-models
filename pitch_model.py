import tensorflow as tf

import pitch_data
import pitch_eval


def pitch_model():
  return tf.estimator.DNNClassifier(
          feature_columns=pitch_data.estimator_cols(),
          hidden_units=[200, 150, 100],
          n_classes=7,
          optimizer=tf.train.AdamOptimizer(),
          dropout=0.1,
          model_dir='model_pitch')


def main(argv):
  model = pitch_model()

  for idx in range(100):
    print('------ TRAIN ----------: {}'.format(idx))
    model.train(
            input_fn=lambda:pitch_data.csv_input_fn('training_data.csv', batchsize=100),
            steps=1000)

    eval_result = model.evaluate(
            input_fn=lambda:pitch_data.csv_eval_fn('test_data.csv', batchsize=100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    if idx % 10 == 0:
      pitch_eval.print_eval(model)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
