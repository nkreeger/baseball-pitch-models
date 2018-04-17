import tensorflow as tf

import pitch_data
import pitch_model


def print_eval(model):
  predictions = model.predict(input_fn=lambda:pitch_data.test_pitch(100))

  template = ('\n** Prediction is "{}" ({:.1f}%), expected "{}"')

  print('\n')

  for pred_dict, expec in zip(predictions, pitch_data.PITCH_CLASSES):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(pitch_data.PITCH_CLASSES[class_id], 100 * probability, expec))

    print('  predictions: {}'.format(list(pred_dict)))
    print('  probabilities:')
    index = 0
    for prob in pred_dict['probabilities']:
      if (prob > 0):
        print('    - {}, {}'.format((100 * prob), pitch_data.PITCH_CLASSES[index]))
      index = index +1

  print('\n')


def main(argv):
  print_eval(pitch_model.pitch_model())


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
