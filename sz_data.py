import tensorflow as tf
import numpy as np


# CSV column definitions
csv_column_types = [
    [],   # px (0)
    [],   # pz (1)
    [],   # sz_top (2)
    [],   # sz_bot (3)
    [0],  # left_handed_batter (4)
    [0],  # label (5)
]


SZ_CLASSES = ['STRIKE', 'BALL']


PX_MIN = -2.65170604056843
PX_MAX = 2.842899614
PZ_MIN = -2.01705841594049
PZ_MAX = 6.06644249133382
SZ_TOP_MIN = 2.85
SZ_TOP_MAX = 4.241794863019148
SZ_BOT_MIN = 1.248894636863092
SZ_BOT_MAX = 3.2130980270561516


def col_keys():
    return [
        'px',
        'pz',
        'sz_top',
        'sz_bot',
        'left_handed_batter'
    ]


def load_np_data(filename):
    with open(filename) as f:
        content = f.readlines()

    features = []
    labels = []
    for line in content:
        items = line.split(',')

        f = [float(x) for x in items[:5]]
        features.append(f)

        labels.append(int(items[5]))

    combo = list(zip(features, labels))
    np.random.shuffle(combo)

    features, labels = zip(*combo)
    return (np.array(features, dtype=np.float32), _to_one_hot(labels, 2))


def _to_one_hot(indices, num_classes):
    one_hot = np.zeros([len(indices), num_classes], dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1
    return one_hot


def estimator_cols():
    return [
        tf.feature_column.numeric_column(
            key='px',
            normalizer_fn=lambda x: ((x - PX_MIN) / (PX_MAX - PX_MIN))),
        tf.feature_column.numeric_column(
            key='pz',
            normalizer_fn=lambda x: ((x - PZ_MIN) / (PZ_MAX - PZ_MIN))),
        tf.feature_column.numeric_column(
            key='sz_top',
            normalizer_fn=lambda x: ((x - SZ_TOP_MIN) / (SZ_TOP_MAX - SZ_TOP_MIN))),
        tf.feature_column.numeric_column(
            key='sz_bot',
            normalizer_fn=lambda x: ((x - SZ_BOT_MIN) / (SZ_BOT_MAX - SZ_BOT_MIN))),
        tf.feature_column.numeric_column(key='left_handed_batter')
    ]


def normalize(x, min, max):
    return (x - min) / (max - min)


def decode_csv_est(line):
    parsed_line = tf.decode_csv(line, record_defaults=csv_column_types)

    px = parsed_line[0]
    pz = parsed_line[1]
    sz_top = parsed_line[2]
    sz_bot = parsed_line[3]
    left_handed_batter = parsed_line[4]

    features = dict(zip(col_keys(), [
        px,
        pz,
        sz_top,
        sz_bot,
        left_handed_batter
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
        'sz_bot': [1.4077442676339658, 1.543131267207734],
        'left_handed_batter': [0, 1]
    }
    return tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
