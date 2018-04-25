#!/bin/sh

# Older CSV data:
#TRAINING_DATA="https://storage.googleapis.com/mlb-pitch-data/training_data.csv.gz"
#TEST_DATA="https://storage.googleapis.com/mlb-pitch-data/test_data.csv.gz"
#
#curl -L $TRAINING_DATA | gunzip > training_data.csv
#curl -L $TEST_DATA | gunzip > test_data.csv

# JSON data
PITCH_TYPE_TRAINING_DATA="https://storage.googleapis.com/mlb-pitch-data/pitch_type_training_data.json.gz"
PITCH_TYPE_TEST_DATA="https://storage.googleapis.com/mlb-pitch-data/pitch_type_test_data.json.gz"
STRIKE_ZONE_TRAINING_DATA="https://storage.googleapis.com/mlb-pitch-data/strike_zone_training_data.json.gz"
STRIKE_ZONE_TEST_DATA="https://storage.googleapis.com/mlb-pitch-data/strike_zone_test_data.json.gz"

curl -L $PITCH_TYPE_TRAINING_DATA | gunzip > pitch_type_training_data.json
curl -L $PITCH_TYPE_TEST_DATA | gunzip > pitch_type_test_data.json
curl -L $STRIKE_ZONE_TRAINING_DATA | gunzip > strike_zone_training_data.json
curl -L $STRIKE_ZONE_TEST_DATA | gunzip > strike_zone_test_data.json

