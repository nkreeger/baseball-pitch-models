import tensorflow as tf

import pitch_data
import pitch_eval
import pitch_model


def main(argv):
  model = pitch_model.model('models')

  for idx in range(10000):
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
  # run_eager(argv=sys.argv)
  tf.app.run(main)
