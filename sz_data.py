import tensorflow as tf


# CSV column definitions
csv_column_types = [
  [],   # px (0)
  [],   # pz (1)
  [],   # sz_top (2)
  [],   # sz_bot (3)
  [0],   # label (4)
]


SZ_CLASSES = ['STRIKE', 'BALL']


PX_MIN = -5.08314439875463
PX_MAX = 4.252838549
PZ_MIN = -3.14564239312703
PZ_MAX = 6.06644249133382
SZ_TOP_MIN = 2.6534936194861984
SZ_TOP_MAX = 4.43416657143635
SZ_BOT_MIN = 1.068002693284189
SZ_BOT_MAX = 2.434942748330954


def col_keys():
  return [
    'px',
    'pz',
    'sz_top',
    'sz_bot',
  ]


def estimator_cols():
  return [
    tf.feature_column.numeric_column(
      key='px'),
      # normalizer_fn=lambda x: ((x - PX_MIN) / (PX_MAX - PX_MIN))),
    tf.feature_column.numeric_column(
      key='pz'),
      # normalizer_fn=lambda x: ((x - PZ_MIN) / (PZ_MAX - PZ_MIN))),
    tf.feature_column.numeric_column(
      key='sz_top'),
      # normalizer_fn=lambda x: ((x - SZ_HEIGHT_MIN) / (SZ_HEIGHT_MAX - SZ_HEIGHT_MIN))),
    tf.feature_column.numeric_column(
      key='sz_bot'),
  ]


def normalize(x, min, max):
  return (x - min) / (max - min)


def decode_csv_est(line):
  parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)

  px = normalize(parsed_line[0], PX_MIN, PX_MAX)
  pz = normalize(parsed_line[1], PZ_MIN, PZ_MAX)
  sz_top = normalize(parsed_line[2], SZ_TOP_MIN, SZ_TOP_MAX)
  sz_bot = normalize(parsed_line[3], SZ_BOT_MIN, SZ_BOT_MAX)

  features = dict(zip(col_keys(), [
    px,
    pz,
    sz_top,
    sz_bot
  ]))

  return features, parsed_line[4]


def csv_input_fn(filename, batchsize=100):
  dataset = tf.data.TextLineDataset([filename])
  dataset = dataset.map(decode_csv_est)
  dataset = dataset.shuffle(20000).repeat().batch(batchsize)
  return dataset


def csv_eval_fn(filename, batchsize=100):
  dataset = tf.data.TextLineDataset([filename])
  dataset = dataset.map(decode_csv_est)
  dataset = dataset.shuffle(batchsize).batch(batchsize)
  return dataset


def predict_input(batch_size):
  samples = {
    'px': [-0.21852914, 1.245771514],
    'pz': [2.059048211, 0.699277829],
    'sz_top': [3.054163756608983, 3.2710515827025475],
    'sz_bot': [1.4077442676339658, 1.543131267207734]
  }
  return tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)

