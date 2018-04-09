#!/bin/sh

TRAINING_DATA="https://storage.googleapis.com/mlb-pitch-data/training_data.csv.gz"
TEST_DATA="https://storage.googleapis.com/mlb-pitch-data/test_data.csv.gz"

curl -L $TRAINING_DATA | gunzip > training_data.csv
curl -L $TEST_DATA | gunzip > test_data.csv
