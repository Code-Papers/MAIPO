#! /usr/bin/env bash

echo "run game 5 ..."
python3 train.py --scenario "simple_reference" --exp-name "simple_reference" --clip-range 1e-4 --save-dir "./policy/" --plots-dir "./learning_curves5/"
echo "run game 6 ..."
python3 train.py --scenario "simple_speaker_listener" --exp-name "simple_speaker_listener" --clip-range 1e-4 --save-dir "./policy/" --plots-dir "./learning_curves5/"
echo "run game 7 ..."
python3 train.py --scenario "simple_spread" --exp-name "simple_spread" --clip-range 1e-4 --save-dir "./policy/" --plots-dir "./learning_curves5/"
