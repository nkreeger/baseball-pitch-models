import tensorflow as tf

import sz_data


def sz_model():
  return tf.estimator.DNNClassifier(
          feature_columns=sz_data.estimator_cols(),
          hidden_units=[20, 20],
          n_classes=2,
          model_dir='model_sz')


def predict(model):
  expected = ['STRIKE', 'BALL']

  predictions = model.predict(input_fn=lambda:sz_data.predict_input(100))
  print expected

  template = ('\n** Prediction is "{}" ({:.1f}%), expected "{}"')
  print('\n')
  for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(sz_data.SZ_CLASSES[class_id], 100 * probability, expec))

    print('  probabilities:')
    index = 0
    for prob in pred_dict['probabilities']:
      if (prob > 0):
        print('    - {}, {}'.format((100 * prob), sz_data.SZ_CLASSES[index]))
      index = index + 1


def main(argv):
  model = sz_model()

  for idx in range(100):
    print('------ TRAIN ----------: {}'.format(idx))
    model.train(
            input_fn=lambda:sz_data.csv_input_fn('sz_training_data.csv', batchsize=10),
            steps=1000)

    eval_result = model.evaluate(
            input_fn=lambda:sz_data.csv_eval_fn('sz_training_data.csv', batchsize=100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    if idx % 10 == 0:
      predict(model)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
