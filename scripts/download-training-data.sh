#!/bin/sh

JULY_2017_DATA="https://storage.googleapis.com/mlb-pitch-data/july_2017_pitches.csv.gz"

target=$JULY_2017_DATA
curl -L $target | gunzip > training_data.csv
