# baseball-pitch-models
Deep TensorFlow models for classifiying pitches and strike zone probabilities.

## Download Training Data:
```
$ ./scripts/download-training-data.sh
```

## Run Pitch Classifier Training Model:
```
$ python pitch_model.py
```

## Run StrikeZone Classifier Training Model:
```
$ python sz_model.py
```

## Start Tensorboard:
- Pitch Classifier:
```
$ tensorboard --logdir=models_pitch/
```

- StrikeZone Classifier:
```
$ tensorboard --logdir=models_sz/
```

## Wipe Models Directory:
```
$ rm -rf models_*
```
