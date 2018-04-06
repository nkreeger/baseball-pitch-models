#!/bin/sh

JUNE_2017_DATA="https://storage.googleapis.com/mlb-pitch-data/june_2017_pitches.csv.gz"
JULY_2017_DATA="https://storage.googleapis.com/mlb-pitch-data/july_2017_pitches.csv.gz"

curl -L $JUNE_2017_DATA | gunzip > training_data.csv
curl -L $JULY_2017_DATA | gunzip > test_data.csv
