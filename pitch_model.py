import keras
import numpy as np
import tensorflowjs as tfjs
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


def keras_model():
  features, labels = pitch_data.load_np_data('pitch_type_training_data.json.csv')

  model = keras.models.Sequential()
  model.add(keras.layers.Dense(
    250, input_shape=[8], use_bias=True, activation='relu', name='Dense1'))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(
    175, use_bias=True, activation='relu', name='Dense2'))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(
    150, use_bias=True, activation='relu', name='Dense3'))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(
    7, use_bias=True, activation='softmax', name='DenseSOFTMAX'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  model.fit(features, labels, batch_size=50, epochs=500)

  # Run prediction on the training set.
  pred_ys = np.argmax(model.predict(features), axis=1)
  true_ys = np.argmax(labels, axis=1)
  final_train_accuracy = np.mean((pred_ys == true_ys).astype(np.float32))
  print('Accuracy on the training set: %g' % final_train_accuracy)

  # tfjs.converters.save_keras_model(model, 'test.keras')


# def main(argv):
#   model = pitch_model()

#   for idx in range(100):
#     print('------ TRAIN ----------: {}'.format(idx))
#     model.train(
#             input_fn=lambda:pitch_data.csv_input_fn('training_data.csv', batchsize=100),
#             steps=1000)

#     eval_result = model.evaluate(
#             input_fn=lambda:pitch_data.csv_eval_fn('test_data.csv', batchsize=100))

#     print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#     if idx % 10 == 0:
#       pitch_eval.print_eval(model)


if __name__ == '__main__':
  keras_model()
  # tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(main)
