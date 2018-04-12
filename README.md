# mlb-pitch-model
Trains a simple classification model to predict MLB PitchFX pitches.

## Download Training Data:
```
$ ./scripts/download-training-data.sh
```

## Run Training Model:
```
$ python pitch_train.py
```

## Start Tensorboard:
```
$ tensorboard --logdir=models/
```

## Wipe Models Directory:
```
$ rm -rf models/*
```
