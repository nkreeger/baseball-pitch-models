import keras
import numpy as np
import tensorflowjs as tfjs
import tensorflow as tf
import sz_data

def sz_model():
  return tf.estimator.DNNClassifier(
          feature_columns=sz_data.estimator_cols(),
          hidden_units=[25, 25],
          n_classes=2,
          optimizer=tf.train.AdamOptimizer(),
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


def keras_model():
  features, labels = sz_data.load_np_data('strike_zone_training_data.json.csv')

  model = keras.models.Sequential()
  model.add(keras.layers.Dense(
    25, input_shape=[5], use_bias=True, activation='relu', name='Dense1'))
  model.add(keras.layers.Dense(
    2, use_bias=True, activation='softmax', name='Dense3'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  model.fit(features, labels, batch_size=50, epochs=100)

  # Run prediction on the training set.
  pred_ys = np.argmax(model.predict(features), axis=1)
  true_ys = np.argmax(labels, axis=1)
  final_train_accuracy = np.mean((pred_ys == true_ys).astype(np.float32))
  print('Accuracy on the training set: %g' % final_train_accuracy)

  tfjs.converters.save_keras_model(model, 'test.keras')


def dnn():
  model = sz_model()

  for idx in range(1):
    print('------ TRAIN ----------: {}'.format(idx))
    model.train(
            input_fn=lambda:sz_data.csv_input_fn('strike_zone_training_data.json.csv', batchsize=100),
            steps=1000)

    eval_result = model.evaluate(
            input_fn=lambda:sz_data.csv_eval_fn('strike_zone_test_data.json.csv', batchsize=100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    if idx % 10 == 0:
      predict(model)


# def main(argv):
#   dnn()


if __name__ == '__main__':
  keras_model()
  # tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(main)
